# Copyright (c) 2025 Alibaba PAI and Nvidia Megatron-LM Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Some MPT Implementation copy from: https://github.com/FlagOpen/FlagScale

import copy
from typing import Literal, Optional
import torch
from torch import Tensor

from collections import OrderedDict
from megatron.core import InferenceParams, tensor_parallel, parallel_state
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.spec_utils import ModuleSpec


from .transformer_config import DeepSeekV3TransformerConfig
from .multi_token_predictor import (
    DeepSeekMultiTokenPredictor,
    roll_tensor,
)


class DeepSeekV3Model(GPTModel):
    """DeepSeek-V3 language model.

    Args:
        config (TransformerConfig):
            Transformer config
        transformer_layer_spec (ModuleSpec):
            Specifies module to use for transformer layers
        vocab_size (int):
            Vocabulary size
        max_sequence_length (int):
            maximum size of sequence. This is used for positional embedding
        pre_process (bool, optional):
            Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional):
            Include an output layer (used with pipeline parallelism). Defaults to True.
        fp16_lm_cross_entropy (bool, optional):
            Defaults to False.
        parallel_output (bool, optional):
            Do not gather the outputs, keep them split across tensor
            parallel ranks. Defaults to True.
        share_embeddings_and_output_weights (bool, optional):
            When True, input embeddings and output logit weights are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope], optional):
            Position embedding type.. Defaults to 'learned_absolute'.
        rotary_percent (float, optional):
            Percent of rotary dimension to use for rotary position embeddings.
            Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional):
            Base period for rotary position embeddings. Ignored unless
            position_embedding_type is 'rope'.
            Defaults to 10000.
        rope_scaling (bool, optional): Toggle RoPE scaling.
        rope_scaling_factor (float): RoPE scaling factor. Default 8.
        scatter_embedding_sequence_parallel (bool, optional):
            Whether embeddings should be scattered across sequence parallel
            region or not. Defaults to True.
        seq_len_interpolation_factor (Optional[float], optional):
            scale of linearly interpolating RoPE for longer sequences.
            The value must be a float larger than 1.0. Defaults to None.
    """

    def __init__(
            self,
            config: DeepSeekV3TransformerConfig,
            transformer_layer_spec: ModuleSpec,
            vocab_size: int,
            max_sequence_length: int,
            pre_process: bool = True,
            post_process: bool = True,
            fp16_lm_cross_entropy: bool = False,
            parallel_output: bool = True,
            share_embeddings_and_output_weights: bool = False,
            position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
            rotary_percent: float = 1.0,
            rotary_base: int = 10000,
            rope_scaling: bool = False,
            rope_scaling_factor: float = 8.0,
            scatter_embedding_sequence_parallel: bool = True,
            seq_len_interpolation_factor: Optional[float] = None,
    ) -> None:

        super().__init__(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            rope_scaling=rope_scaling,
            rope_scaling_factor=rope_scaling_factor,
            scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
        )

        self.use_multi_token_prediction = config.use_multi_token_prediction

        if self.use_multi_token_prediction and self.post_process:
            # init mtp embeddings
            self.mtp_embedding = LanguageModelEmbedding(
                config=config,
                vocab_size=vocab_size,
                max_sequence_length=max_sequence_length,
                position_embedding_type=position_embedding_type,
                scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
            )
            # init mtp norm, linar_proj and transformer block
            mtp_config = copy.deepcopy(config)
            mtp_config.pipeline_model_parallel_size = 1
            mtp_config.num_layers = 1
            self.mtp_predictor = DeepSeekMultiTokenPredictor(
                config=mtp_config,
                transformer_layer_spec=transformer_layer_spec,
            )
            # mtp output lm head is the same with main model output layer

        # Mtp embedding shares weight with main model embedding
        # In a pipelined setup with more than one stage, the initial
        # embedding layer and the mtp embedding are on different workers,
        # thus call an all-reduce to ensure that first and last stages have the same initial parameter values
        if self.use_multi_token_prediction and (self.pre_process or self.post_process):
            self.setup_mtp_embeddings()

    def forward(
            self,
            input_ids: Tensor,
            position_ids: Tensor,
            attention_mask: Tensor,
            decoder_input: Tensor = None,
            labels: Tensor = None,
            inference_params: InferenceParams = None,
            packed_seq_params: PackedSeqParams = None,
            extra_block_kwargs: dict = None,
            runtime_gather_output: Optional[bool] = None,
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units

        Args:
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
        """
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        rotary_pos_cos = None
        rotary_pos_sin = None
        if self.position_embedding_type == 'rope' and not self.config.multi_latent_attention:
            if not self.training and self.config.flash_decode and inference_params:
                # Flash decoding uses precomputed cos and sin for RoPE
                rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb_cache.setdefault(
                    inference_params.max_sequence_length,
                    self.rotary_pos_emb.get_cos_sin(inference_params.max_sequence_length),
                )
            else:
                rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                    inference_params, self.decoder, decoder_input, self.config, packed_seq_params
                )
                rotary_pos_emb = self.rotary_pos_emb(
                    rotary_seq_len,
                    packed_seq=packed_seq_params is not None
                               and packed_seq_params.qkv_format == 'thd',
                )
        if (
                (self.config.enable_cuda_graph or self.config.flash_decode)
                and rotary_pos_cos is not None
                and inference_params
        ):
            sequence_len_offset = torch.tensor(
                [inference_params.sequence_len_offset] * inference_params.current_batch_size,
                dtype=torch.int32,
                device=rotary_pos_cos.device,  # Co-locate this with the rotary tensors
            )
        else:
            sequence_len_offset = None

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            **(extra_block_kwargs or {}),
        )

        if not self.post_process:
            return hidden_states

        # logits of main model
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits, _ = self.output_layer(
            hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
        )
        logging_logits = logits

        # logits of mtp predictors
        if self.use_multi_token_prediction:
            # mtp embedding
            decoder_input_mtps = self.mtp_embedding(input_ids=input_ids, position_ids=position_ids)
            # mtp norm, linear proj and transformer block
            hidden_states_mtps = self.mtp_predictor(
                decoder_input=decoder_input_mtps,
                attention_mask=attention_mask,
                pre_hidden_states=hidden_states,
            )
            # mtp output lm head
            logits_mtps = []
            for idx, hidden_states_mtp in enumerate(hidden_states_mtps):
                logits_mtp, _ = self.output_layer(
                    hidden_states_mtp, weight=output_weight, runtime_gather_output=runtime_gather_output
                )
                logits_mtps.append(logits_mtp)
            logging_logits = torch.cat([logits, torch.cat(logits_mtps, dim=1)], dim=1)

        if has_config_logger_enabled(self.config):
            payload = OrderedDict(
                {
                    'input_ids': input_ids,
                    'position_ids': position_ids,
                    'attention_mask': attention_mask,
                    'decoder_input': decoder_input,
                    'logits': logging_logits,
                }
            )
            log_config_to_disk(self.config, payload, prefix='input_and_logits')

        if labels is None:
            logits = logits.transpose(0, 1).contiguous()
            if not self.use_multi_token_prediction:
                return logits
            for idx, logit in enumerate(logits_mtps):
                logits_mtps[idx] = logit.transpose(0, 1).contiguous()
            return [logits, logits_mtps]

        # compute loss
        loss = self.compute_language_model_loss(labels, logits)
        if not self.use_multi_token_prediction:
            return loss
        loss_mtps = self.compute_mtp_predictor_loss(labels, logits_mtps)
        return [loss, loss_mtps]

    def compute_mtp_predictor_loss(self, labels: Tensor, logits_mtps: Tensor):
        roll_labels = labels  # [b s]
        labels_mtps = []

        num_mtps = len(logits_mtps)
        for i in range(num_mtps):
            labels_mtp, _ = roll_tensor(roll_labels, dims=1)
            roll_labels = labels_mtp
            labels_mtp = labels_mtp.transpose(0, 1).contiguous()  # [b s] ==> [s b]
            labels_mtps.append(labels_mtp)

        logits_mtps = torch.cat(logits_mtps, 1)  # [s b h]
        labels_mtps = torch.cat(labels_mtps, 1)  # [s b h]
        losses_mtps = tensor_parallel.vocab_parallel_cross_entropy(logits_mtps.float(), labels_mtps)
        losses_mtps = losses_mtps.transpose(0, 1).contiguous().float()  # [b s]

        return losses_mtps

    def share_embedding_or_mtp_embedding(self) -> Tensor:
        """Gets the emedding weight or mtp embedding weight when share embeddings between main model and mtp modules

        Returns:
            Tensor: During pre processing it returns the input embeddings weight while during post processing it returns the mtp embeddings weight
        """
        if self.pre_process:
            return self.embedding.word_embeddings.weight
        elif self.post_process:
            return self.mtp_embedding.word_embeddings.weight
        return None

    def setup_mtp_embeddings(self) -> None:
        """Sets up embedding layer in first stage and mtp embedding layer in last stage.
        """

        # Set `is_embedding_or_output_parameter` attribute.
        if self.pre_process:
            self.embedding.word_embeddings.weight.is_embedding_or_output_parameter = True
        if self.post_process and self.mtp_embedding.word_embeddings.weight is not None:
            self.mtp_embedding.word_embeddings.weight.is_embedding_or_output_parameter = True

        if parallel_state.get_pipeline_model_parallel_world_size() == 1:
            # Zero out wgrad if sharing embeddings between two layers on same
            # pipeline stage to make sure grad accumulation into main_grad is
            # correct and does not include garbage values (e.g., from torch.empty).
            self.share_embedding_or_mtp_embedding().zero_out_wgrad = True
            return

        if parallel_state.is_pipeline_first_stage() and self.pre_process and not self.post_process:
            self.share_embedding_or_mtp_embedding().shared_embedding = True

        if self.post_process and not self.pre_process:
            assert not parallel_state.is_pipeline_first_stage()
            # set word_embeddings weights to 0 here, then copy first
            # stage's weights using all_reduce below.
            self.mtp_embedding.word_embeddings.weight.data.fill_(0)
            self.mtp_embedding.word_embeddings.weight.shared = True
            self.mtp_embedding.word_embeddings.weight.shared_embedding = True

        # Parameters are shared between the word embeddings layers, and the
        # mtp embeddings at the end of the model. In a pipelined setup with more than
        # one stage, the initial embedding layer and the mtp embedding are on different
        # workers, so we do the following:
        # 1. Create a second copy of word_embeddings on the last stage, with
        #    initial parameters of 0.0.
        # 2. Do an all-reduce between the first and last stage to ensure that
        #    the two copies of word_embeddings start off with the same
        #    parameter values.
        # 3. In the training loop, before an all-reduce between the grads of
        #    the two word_embeddings layers to ensure that every applied weight
        #    update is the same on both stages.

        # Ensure that first and last stages have the same initial parameter
        # values.
        if torch.distributed.is_initialized():
            if parallel_state.is_rank_in_embedding_group():
                weight = self.share_embedding_or_mtp_embedding()
                weight.data = weight.data.cuda()
                embedding_group = parallel_state.get_embedding_group()
                if not isinstance(embedding_group, list):
                    torch.distributed.all_reduce(
                        weight.data, group=parallel_state.get_embedding_group()
                    )
                else:
                    original_weight = weight.clone().detach().data
                    for group in embedding_group:
                        weight.data.copy_(original_weight)
                        torch.distributed.all_reduce(weight.data, group=group)

        elif not getattr(DeepSeekV3Model, "embedding_warning_printed", False):
            logging.getLogger(__name__).warning(
                "Distributed processes aren't initialized, so the mtp embeddings "
                "is not initialized with weights from the word embeddings. "
                "If you are just manipulating a model this is fine, but "
                "this needs to be handled manually. If you are training "
                "something is definitely wrong."
            )
            DeepSeekV3Model.embedding_warning_printed = True
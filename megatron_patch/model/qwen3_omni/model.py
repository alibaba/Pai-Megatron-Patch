# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import logging
from typing import Optional

import torch
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig

from megatron.core import InferenceParams
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.packed_seq_params import PackedSeqParams

from .transformer_config import Qwen3OmniTransformerConfig
from .visionmodel import Qwen3VisionModel
from .gpt_model import GPTModel


# Note: This is under development and may be missing features.
class Qwen3OmniModel(MegatronModule):
    """Qwen3VL multi-modal model.

    Args:
        language_transformer_config (TransformerConfig): Transformer config for the language model.
        language_transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers of the language model.
        language_vocab_size (int): Language model vocabulary size.
        language_max_sequence_length (int): Language model maximum sequence length. This is used for positional embedding.
        vision_transformer_config (TransformerConfig): Transformer config for the vision model.
        vision_transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers of the vision model.
        drop_vision_class_token (bool): Drop vision class token(s) before input to the language model.
        vision_projection_config (TransformerConfig): Config for the projection from vision model outputs to language model inputs.
        vision_projection_layer_spec (ModuleSpec): Specifies the module to use for the vision projection.
        vision_projection_type (str): Type of the vision projection to use. Default is a 2-layer MLP.
        audio_transformer_config (AutoConfig): HF Transformer config for the audio model.
        allow_missing_vision_projection_checkpoint (bool): Allow vision projection weights to be missing when loading a checkpoint. Default False.
        parallel_output (bool): Do not gather the outputs, keep them split across tensor parallel ranks. This is typically True for training and False for inference.
        language_position_embedding_type (str): Position embedding type to use in the language model. Default learned absolute.
        language_rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings in the language model. Defaults to 1.0.
        pre_process (bool): Include the embedding layer in the gpt decoder (used with pipeline parallelism). Defaults to True.
        post_process (bool): Include an output layer and a layernorm in the gpt decoder (used with pipeline parallelism). Defaults to True.
        add_encoder (bool): Construct the encoder module (used with pipeline parallelism). Defaults to True. When we use pipelining, the encoder
            will live on only a subset of the pipeline stages (specifically, only the first stage).
        add_decoder (bool): Construct the decoder module (used with pipeline parallelism). Defaults to True. When we use pipelining, the decoder
            will live on only a subset of the pipeline stages (specifically, every stage after the first one).
        img_h (int): The height of each image that the ViT will see.
        img_w (int): The width of each image that the ViT will see.
        patch_dim (int): The size of each patch side.
        img_embedding_idx (int): Index in the language_embeddings tensor where image_embeddings should be inserted. Defaults to 0.
    """

    def __init__(
        self,
        language_transformer_config: Qwen3OmniTransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        vision_transformer_config: TransformerConfig,
        vision_transformer_layer_spec: ModuleSpec,
        drop_vision_class_token: bool,
        vision_projection_config: TransformerConfig,
        vision_projection_layer_spec: ModuleSpec,
        vision_projection_type: str = "mlp",
        audio_transformer_config: Qwen3OmniMoeAudioEncoderConfig = None,
        allow_missing_vision_projection_checkpoint: bool = False,
        parallel_output: bool = True,
        language_position_embedding_type: str = 'rope',
        language_rotary_percent: float = 1.0,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        language_rotary_base: int = 10000,
        fp16_lm_cross_entropy: bool = False,
        language_share_embeddings_and_output_weights: bool=False,
        vp_stage: int=None
    ) -> None:
        super().__init__(config=language_transformer_config)

        logging.getLogger(__name__).warning(
            "Qwen2VL model is under development and may be missing features."
        )

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        
        self.encoder_hidden_state = None
        self.vision_model = None
        self.audio_model = None
        self.vision_projection = None
        self.language_model = None

        self.square_merge_size = vision_projection_config.ffn_hidden_size // vision_transformer_config.hidden_size

        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        self.share_embeddings_and_output_weights = False
        if self.pre_process:
            self.vision_model = Qwen3VisionModel(
                vision_transformer_config,
                vision_transformer_layer_spec,
                vision_projection_config,
                vision_projection_layer_spec,
                projection_type=vision_projection_type,
                pre_process=True,
                post_process=True
            )

            self.audio_model = Qwen3OmniMoeAudioEncoder._from_config(audio_transformer_config)


        self.language_model = GPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_vocab_size,
            max_sequence_length=language_max_sequence_length,
            parallel_output=parallel_output,
            position_embedding_type=language_position_embedding_type,
            rotary_percent=language_rotary_percent,
            pre_process=self.pre_process,
            post_process=self.post_process,
            rotary_base=language_rotary_base,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            share_embeddings_and_output_weights=language_share_embeddings_and_output_weights,
            rope_scaling=False,
            mtp_block_spec=None,
            vp_stage=vp_stage,
        )
        self.share_embeddings_and_output_weights = (
            self.language_model.share_embeddings_and_output_weights
        )

    def shared_embedding_or_output_weight(self):
        """This is a convenience method to surface the language model's word embeddings, which is
        necessary for `finalize_model_grads._allreduce_word_embedding_grads`."""
        if self.add_decoder:
            return self.language_model.shared_embedding_or_output_weight()
        return None

    def set_input_tensor(self, input_tensor) -> None:
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for Qwen2VL'
        
        if self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self, freeze_language_model: bool, freeze_vision_model: bool, freeze_vision_projection: bool
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False for the module's parameters.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_vision_projection (bool): Freeze the vision projection module.
        """
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)
        if freeze_vision_projection and self.vision_projection is not None:
            modules.append(self.vision_projection)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
    ):
        """
        Encodes audios into continuous embeddings that can be forwarded to the language model.

        Args:
            input_features (`torch.FloatTensor`):
                The tensors corresponding to the input audios.
            feature_attention_mask (`torch.LongTensor`, *optional*):
                Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:
            audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
        """
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            audio_feature_lengths = None

        feature_lens = audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
        #RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same
        audio_outputs = self.audio_model(
            input_features.bfloat16(),
            feature_lens=feature_lens,
        )
        
        audio_features = audio_outputs.last_hidden_state

        return audio_features

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        vision_data: torch.Tensor = None,
        vision_grid_thw: torch.Tensor = None,
        audio_data: torch.Tensor = None,
        audio_lengths: torch.Tensor = None,
        video_start_index: int = -1,
        image_input_mask: torch.Tensor = None,
        video_input_mask: torch.Tensor = None,
        audio_input_mask: torch.Tensor = None,
        audio_feature_attention_mask: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
    ) -> torch.Tensor:
        """Forward function of the Qwen2VL model.

        Args:
            image_data (torch.Tensor): input image of shape [total_thw_size, n_features].
            input_ids (torch.Tensor): input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): attention mask for the language model [batch, 1, combined_seq_len, combined_seq_len].
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            inference_params (InferenceParams): Inference-time parameters including KV cache.

            video_start_index:
                0 -- all video
                len(video_seq) -- all image
                others -- mixture
            *_input_mask: should not be None in the first PP stage
        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape [b, s, vocab_size].
        """
        use_inference_kv_cache = (
            inference_params is not None
            and "image_tokens_count" in inference_params.key_value_memory_dict
        )
        if use_inference_kv_cache:
            raise NotImplementedError()
        
        if self.pre_process:
            vision_embeds = None
            deepstack_feature_lists = None
            if vision_grid_thw.shape[0] > 0:
                vision_embeds, deepstack_feature_lists = self.vision_model(
                    vision_data=vision_data, # If None, vision model should use intermediate outputs (EPP > 1)
                    grid_thw=vision_grid_thw # should provided in each EPP stage
                )

            audio_embds = None
            if audio_data is not None:
                attention_mask_length = audio_data.shape[-1]
                #HF: audio_data: torch.Size([2, 128, 403]), audio_feature_lengths: torch.Size([2, 403])
                #HF: output: torch.Size([78, 2048])
                #Mcore audio_data: torch.Size([1, 80, 1209]), audio_feature_lengths: torch.Size([1, 1209])
                audio_embeds = self.get_audio_features(
                    audio_data,
                    feature_attention_mask=audio_feature_attention_mask[:, :attention_mask_length],
                    audio_feature_lengths=None,
                )

            # If running inference, the language model KV cache will be updated for image token positions.
            # Here we store the image tokens sequence length, which can be used as an offset to the KV cache later.
            if inference_params is not None:
                raise NotImplementedError()
                # inference_params.key_value_memory_dict["image_tokens_count"] = (
                #     vision_embeddings.shape[0]
                # )
            
            # If running inference, we can skip image token computation if they were computed already earlier for this sample.
            if use_inference_kv_cache:
                language_embeddings: torch.Tensor = self.language_model.embedding(
                    input_ids=input_ids,
                    position_ids=None # NOTE: disable
                )  # [text_seq_len, b, h_language]
                # NOTE: why not cat here? is it the combined embeddings useless?
                combined_embeddings = language_embeddings
            elif vision_embeds is not None:
                if image_input_mask is not None:
                    image_input_mask = image_input_mask.T # shape [seqlen, mbs]
                if video_input_mask is not None:
                    video_input_mask = video_input_mask.T

                if video_start_index == 0:
                    image_embeds = None
                    video_embeds = vision_embeds
                    visual_pos_masks = video_input_mask 
                elif video_start_index == vision_embeds.shape[0]:
                    image_embeds = vision_embeds
                    video_embeds = None
                    visual_pos_masks = image_input_mask
                elif 0 < video_start_index < vision_embeds.shape[0]:
                    image_embeds = vision_embeds[:video_start_index]
                    video_embeds = vision_embeds[video_start_index:]
                    visual_pos_masks = torch.logical_or(image_input_mask, video_input_mask)
                else:
                    raise ValueError(f"Expect video token start index in range [0, {vision_embeds.shape[0]}], but got {video_start_index}")
                

                combined_embeddings = self.language_model.embedding(
                    input_ids=input_ids,
                    position_ids=None, # NOTE: disable
                    image_input_mask=image_input_mask,
                    video_input_mask=video_input_mask,
                    image_embeds=image_embeds,
                    video_embeds=video_embeds,
                    audio_embeds=audio_embeds,
                )  # [text_seq_len, b, h_language]
            else:
                combined_embeddings = self.language_model.embedding(
                    input_ids=input_ids,
                    position_ids=None # NOTE: disable
                )  # [text_seq_len, b, h_language]
            
        else:
            combined_embeddings = None
            visual_pos_masks = None
            deepstack_feature_lists = None


        

        output = self.language_model(
            input_ids=None,
            position_ids=position_ids,              # None in encoder
            attention_mask=attention_mask,          # None in encoder
            decoder_input=combined_embeddings,      # only not None in the first decoder PP stage
            labels=labels,                          # only not None in the last decoder PP stage
            inference_params=inference_params,      # currently always None
            packed_seq_params=packed_seq_params,    # currently always None
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_feature_lists,
            **(extra_block_kwargs or {}),
        )
        return output

    # Slightly modified from Qwen3VLTextModel.get_rope_index
    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""
        #! =======START===========
        # Since we use timestamps to seperate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        #! =======END===========
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas


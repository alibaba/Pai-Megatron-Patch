# Copyright (c) 2023 Alibaba PAI and Nvidia Megatron-LM Team.
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

import torch
import math
from typing import Optional

from megatron import get_args
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.model.enums import AttnMaskType
from megatron.model.enums import LayerType
from megatron.model.module import MegatronModule
from megatron.model.utils import get_linear_layer
from megatron.model.utils import init_method_normal
from megatron.model.utils import scaled_init_method_normal

from .transformer import ParallelTransformer
from .positional_embeddings import RotaryEmbedding

def _get_interleave(n):
    def _get_interleave_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            _get_interleave_power_of_2(closest_power_of_2)
            + _get_interleave(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


def _fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def _buffered_future_mask(tensor, maxpos, alibi, attn_heads):
    _future_mask = torch.triu(_fill_with_neg_inf(torch.zeros([maxpos, maxpos])), 1)
    _future_mask = _future_mask.unsqueeze(0) + alibi
    new_future_mask = _future_mask.to(tensor)
    return new_future_mask[: tensor.shape[0] * attn_heads, :maxpos, :maxpos]


def _gen_alibi_mask(tensor, n_head, max_pos):
    slopes = torch.Tensor(_get_interleave(n_head))
    position_point = torch.arange(max_pos) - max_pos + 1
    position_point = position_point.unsqueeze(0).unsqueeze(0).expand(n_head, -1, -1)
    diag = torch.diag(position_point[0])
    position_point = position_point - diag.unsqueeze(0).unsqueeze(0).transpose(-1, -2)
    alibi = slopes.unsqueeze(1).unsqueeze(1) * position_point
    alibi = alibi.view(n_head, 1, max_pos)
    alibi_mask = torch.triu(_fill_with_neg_inf(torch.zeros([max_pos, max_pos])), 1)
    alibi_mask = alibi_mask.unsqueeze(0) + alibi
    return alibi_mask


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    if len(mask.size()) == 3:
        bsz, src_len, _ = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len
        expanded_mask = mask[:,None,:,:].expand(bsz, 1, tgt_len, src_len).to(dtype)
    else:
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len
        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def parallel_lm_logits(input_, word_embeddings_weight, parallel_output,
                       bias=None):
    """LM logits using word embedding weights."""
    args = get_args()
    # Parallel logits.
    if args.async_tensor_model_parallel_allreduce or\
            args.sequence_parallel:
        input_parallel = input_
        model_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        async_grad_allreduce = args.async_tensor_model_parallel_allreduce and \
            model_parallel and not args.sequence_parallel
    else:
        input_parallel = tensor_parallel.copy_to_tensor_model_parallel_region(input_)
        async_grad_allreduce = False

    # Matrix multiply.
    logits_parallel = tensor_parallel.linear_with_grad_accumulation_and_async_allreduce(
        input=input_parallel,
        weight=word_embeddings_weight,
        bias=bias,
        gradient_accumulation_fusion=args.gradient_accumulation_fusion,
        async_grad_allreduce=async_grad_allreduce,
        sequence_parallel=args.sequence_parallel)
    # Gather if needed.

    if parallel_output:
        return logits_parallel

    return tensor_parallel.gather_from_tensor_model_parallel_region(logits_parallel)


def get_language_model(config, num_tokentypes, add_pooler,
                       encoder_attn_mask_type,
                       add_encoder=True,
                       add_decoder=False,
                       decoder_attn_mask_type=AttnMaskType.causal,
                       pre_process=True, post_process=True):
    """Build language model and return along with the key to save."""
    args = get_args()
    if config.init_method is None:
        config.init_method = init_method_normal(config.init_method_std)

    if config.output_layer_init_method is None:
        config.output_layer_init_method = scaled_init_method_normal(config.init_method_std,
                                                                    config.num_layers)

    # Language model.
    language_model = TransformerLanguageModel(
        config,
        encoder_attn_mask_type,
        num_tokentypes=num_tokentypes,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
        decoder_attn_mask_type=decoder_attn_mask_type,
        add_pooler=add_pooler,
        pre_process=pre_process,
        post_process=post_process
    )
    # key used for checkpoints.
    language_model_key = 'language_model'

    return language_model, language_model_key


class Pooler(MegatronModule):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(self, hidden_size, init_method):
        super(Pooler, self).__init__()
        args = get_args()
        self.dense = get_linear_layer(hidden_size, hidden_size, init_method)
        self.sequence_parallel = args.sequence_parallel


    def forward(self, hidden_states, sequence_index=0):
        # hidden_states: [s, b, h]
        # sequence_index: index of the token to pool.

        # gather data along sequence dimensions
        # same pooler is run on all tensor parallel nodes
        if self.sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=False)

        pooled = hidden_states[sequence_index, :, :]
        pooled = self.dense(pooled)
        pooled = torch.tanh(pooled)
        return pooled


class Embedding(MegatronModule):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
        embedding_weights_in_fp32: casts word embedding weights to
                                   fp32 before sampling. Required to
                                   maintain reproducibility when
                                   training in bf16.
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 max_sequence_length,
                 embedding_dropout_prob,
                 config,
                 num_tokentypes=0,
                 embedding_weights_in_fp32=False):
        super(Embedding, self).__init__()

        self.hidden_size = hidden_size
        self.init_method = config.init_method
        self.num_tokentypes = num_tokentypes

        args = get_args()

        # Word embeddings (parallel).
        self.embedding_weights_in_fp32 = embedding_weights_in_fp32
        self.params_dtype = args.params_dtype
        self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
            vocab_size, self.hidden_size, config=config, init_method=config.init_method)
        self._word_embeddings_key = 'word_embeddings'

        # Position embedding (serial).
        self.add_position_embedding = args.position_embedding_type == 'learned_absolute'
        if self.add_position_embedding:
            self.position_embeddings = torch.nn.Embedding(
                max_sequence_length, self.hidden_size)
            self._position_embeddings_key = 'position_embeddings'
            # Initialize the position embeddings.
            if args.perform_initialization:
                self.init_method(self.position_embeddings.weight)

        # Token type embedding.
        # Add this as an optional field that can be added through
        # method call so we can load a pretrain model without
        # token types and add them as needed.
        self._tokentype_embeddings_key = 'tokentype_embeddings'
        if self.num_tokentypes > 0:
            self.tokentype_embeddings = torch.nn.Embedding(self.num_tokentypes,
                                                           self.hidden_size)
            # Initialize the token-type embeddings.
            if args.perform_initialization:
                self.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        self.fp32_residual_connection = args.fp32_residual_connection
        self.sequence_parallel = args.sequence_parallel
        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

    def zero_parameters(self):
        """Zero out all parameters in embedding."""
        self.word_embeddings.weight.data.fill_(0)
        self.word_embeddings.weight.shared = True
        if self.add_position_embedding:
            self.position_embeddings.weight.data.fill_(0)
            self.position_embeddings.weight.shared = True
        if self.num_tokentypes > 0:
            self.tokentype_embeddings.weight.data.fill_(0)
            self.tokentype_embeddings.weight.shared = True

    def add_tokentype_embeddings(self, num_tokentypes):
        """Add token-type embedding. This function is provided so we can add
        token-type embeddings in case the pretrained model does not have it.
        This allows us to load the model normally and then add this embedding.
        """
        if self.tokentype_embeddings is not None:
            raise Exception('tokentype embeddings is already initialized')
        if torch.distributed.get_rank() == 0:
            print('adding embedding for {} tokentypes'.format(num_tokentypes),
                  flush=True)
        self.num_tokentypes = num_tokentypes
        self.tokentype_embeddings = torch.nn.Embedding(num_tokentypes,
                                                       self.hidden_size)
        # Initialize the token-type embeddings.
        args = get_args()
        self.init_method(self.tokentype_embeddings.weight)

    def forward(self, input_ids, position_ids, tokentype_ids=None):
        # Embeddings.
        if self.embedding_weights_in_fp32:
            self.word_embeddings = self.word_embeddings.to(torch.float32)
        words_embeddings = self.word_embeddings(input_ids)
        if self.embedding_weights_in_fp32:
            words_embeddings = words_embeddings.to(self.params_dtype)
            self.word_embeddings = self.word_embeddings.to(self.params_dtype)
        if self.add_position_embedding:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = words_embeddings + position_embeddings
        else:
            embeddings = words_embeddings

        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            embeddings = embeddings + self.tokentype_embeddings(tokentype_ids)
        else:
            assert self.tokentype_embeddings is None

        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        embeddings = embeddings.transpose(0, 1).contiguous()

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()

        # Dropout.
        if self.sequence_parallel:
            embeddings = tensor_parallel.scatter_to_sequence_parallel_region(embeddings)
            with tensor_parallel.get_cuda_rng_tracker().fork():
                embeddings = self.embedding_dropout(embeddings)
        else:
            embeddings = self.embedding_dropout(embeddings)

        return embeddings

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        state_dict_[self._word_embeddings_key] \
            = self.word_embeddings.state_dict(prefix=prefix,
                                              keep_vars=keep_vars)
        if self.add_position_embedding:
            state_dict_[self._position_embeddings_key] \
                = self.position_embeddings.state_dict(prefix=prefix,
                                                  keep_vars=keep_vars)
        if self.num_tokentypes > 0:
            state_dict_[self._tokentype_embeddings_key] \
                = self.tokentype_embeddings.state_dict(prefix=prefix,
                                                       keep_vars=keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Word embedding.
        if self._word_embeddings_key in state_dict:
            state_dict_ = state_dict[self._word_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'word_embeddings' in key:
                    state_dict_[key.split('word_embeddings.')[1]] \
                        = state_dict[key]
        self.word_embeddings.load_state_dict(state_dict_, strict=strict)

        # Position embedding.
        if self.add_position_embedding:
            if self._position_embeddings_key in state_dict:
                state_dict_ = state_dict[self._position_embeddings_key]
            else:
                # for backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if 'position_embeddings' in key:
                        state_dict_[key.split('position_embeddings.')[1]] \
                            = state_dict[key]
            self.position_embeddings.load_state_dict(state_dict_, strict=strict)

        # Tokentype embedding.
        if self.num_tokentypes > 0:
            state_dict_ = {}
            if self._tokentype_embeddings_key in state_dict:
                state_dict_ = state_dict[self._tokentype_embeddings_key]
            else:
                # for backward compatibility.
                for key in state_dict.keys():
                    if 'tokentype_embeddings' in key:
                        state_dict_[key.split('tokentype_embeddings.')[1]] \
                            = state_dict[key]
            if len(state_dict_.keys()) > 0:
                self.tokentype_embeddings.load_state_dict(state_dict_,
                                                          strict=strict)
            else:
                print('***WARNING*** expected tokentype embeddings in the '
                      'checkpoint but could not find it', flush=True)


class TransformerLanguageModel(MegatronModule):
    """Transformer language model.

    Arguments:
        transformer_hparams: transformer hyperparameters
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self,
                 config,
                 encoder_attn_mask_type,
                 num_tokentypes=0,
                 add_encoder=True,
                 add_decoder=False,
                 decoder_attn_mask_type=AttnMaskType.causal,
                 add_pooler=False,
                 pre_process=True,
                 post_process=True):
        args = get_args()
        # TODO: passing share_embeddings_and_output_weights=False will not work correctly for T5 and embeddings will not be synced. Fix later for T5.
        if args.untie_embeddings_and_output_weights: assert not add_decoder
        super(TransformerLanguageModel, self).__init__(share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights)

        self.pre_process = pre_process
        self.post_process = post_process
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_tokentypes = num_tokentypes
        self.init_method = config.init_method
        self.add_encoder = add_encoder
        self.encoder_attn_mask_type = encoder_attn_mask_type
        self.add_decoder = add_decoder
        self.decoder_attn_mask_type = decoder_attn_mask_type
        self.add_pooler = add_pooler
        self.encoder_hidden_state = None
        self.add_retriever = args.retro_add_retriever
        self.untie_embeddings_and_output_weights = args.untie_embeddings_and_output_weights
        self.first_run = True
        self.alibi_mask = None
        self.seq_length_with_past = args.seq_length
        self.seq_length = args.seq_length
        self.use_alibi_mask = args.use_alibi_mask
        self.n_head = args.num_attention_heads
        self.max_cache_pos = args.seq_length
        # Embeddings.
        if self.pre_process:
            self.embedding = Embedding(self.hidden_size,
                                       args.padded_vocab_size,
                                       args.max_position_embeddings,
                                       args.hidden_dropout,
                                       config,
                                       self.num_tokentypes,
                                       args.embedding_weights_in_fp32)
            self._embedding_key = 'embedding'

        # Rotary positional embeddings
        self.use_rotary_position_embeddings = \
            args.position_embedding_type == 'rope'
        if self.use_rotary_position_embeddings:
            self.seq_length = args.seq_length
            rotary_dim = args.hidden_size // args.num_attention_heads \
                if args.kv_channels is None else args.kv_channels

            if args.rotary_percent < 1.0:
                rotary_dim = int(rotary_dim * args.rotary_percent)

            # partial rotary embeddings, which is better than full rotary
            # Wang and Komatsuzaki et al
            # https://github.com/kingoflolz/mesh-transformer-jax/
            self.rotary_pos_emb = RotaryEmbedding(
                rotary_dim,
                max_position_embeddings=self.seq_length
            )

        # Encoder (usually set to True, False if part of an encoder-decoder
        # architecture and in encoder-only stage).
        if self.add_encoder:
            self.encoder = ParallelTransformer(
                config,
                model_type=args.model_type if not args.retro_add_retriever \
                    else ModelType.retro_decoder,
                self_attn_mask_type=self.encoder_attn_mask_type,
                pre_process=self.pre_process,
                post_process=self.post_process,
            )
            self._encoder_key = 'encoder'
        else:
            self.encoder = None

        # Decoder (usually set to False, True if part of an encoder-decoder
        # architecture and in decoder-only stage).
        if self.add_decoder:
            self.decoder = ParallelTransformer(
                config,
                model_type=args.model_type,
                layer_type=LayerType.decoder,
                self_attn_mask_type=self.decoder_attn_mask_type,
                pre_process=self.pre_process,
                post_process=self.post_process)
            self._decoder_key = 'decoder'
        else:
            self.decoder = None

        if self.post_process:
            # Pooler.
            if self.add_pooler:
                self.pooler = Pooler(self.hidden_size, self.init_method)
                self._pooler_key = 'pooler'

            if self.untie_embeddings_and_output_weights:
                self.output_layer = tensor_parallel.ColumnParallelLinear(
                    args.hidden_size,
                    args.padded_vocab_size,
                    config=config,
                    init_method=self.init_method,
                    bias=False) # Setting bias to False always to keep it consistent with embedding tying that also does not have a bias.
                self._output_layer_key = 'output_layer'

    def set_input_tensor(self, input_tensor):
        """ See megatron.model.transformer.set_input_tensor()"""

        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        if self.add_encoder and self.add_decoder:
            assert len(input_tensor) == 1, \
                'input_tensor should only be length 1 for stage with both encoder and decoder'
            self.encoder.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            assert len(input_tensor) == 1, \
                'input_tensor should only be length 1 for stage with only encoder'
            self.encoder.set_input_tensor(input_tensor[0])
        elif self.add_decoder:
            if len(input_tensor) == 2:
                self.decoder.set_input_tensor(input_tensor[0])
                self.encoder_hidden_state = input_tensor[1]
            elif len(input_tensor) == 1:
                self.decoder.set_input_tensor(None)
                self.encoder_hidden_state = input_tensor[0]
            else:
                raise Exception('input_tensor must have either length 1 or 2')
        else:
            raise Exception('Stage must have at least either encoder or decoder')

    def get_alibi_mask(self, tensor, seq_length_with_past):
        if self.training:
            slopes = torch.Tensor(_get_interleave(self.n_head))
            position_point = (
                torch.arange(seq_length_with_past) - seq_length_with_past + 1
            )
            position_point = (
                position_point.unsqueeze(0)
                .unsqueeze(0)
                .expand(self.n_head, seq_length_with_past, -1)
            )
            diag = torch.diag(position_point[0])
            position_point = position_point - diag.unsqueeze(0).unsqueeze(0).transpose(
                -1, -2
            )
            alibi = slopes.unsqueeze(1).unsqueeze(1) * position_point
            mask = _buffered_future_mask(
                tensor, seq_length_with_past, alibi, self.n_head
            )
        else:
            if self.first_run:
                self.first_run = False
                self.register_buffer(
                    "future_mask",
                    _gen_alibi_mask(tensor, self.n_head, self.max_cache_pos).to(
                        tensor
                    ),
                    persistent=False,
                )
            if seq_length_with_past > self.max_cache_pos:
                self.max_cache_pos = seq_length_with_past
                self.register_buffer(
                    "future_mask",
                    _gen_alibi_mask(tensor, self.n_head, self.max_cache_pos).to(
                        tensor
                    ),
                    persistent=False,
                )
            mask = self.future_mask[
                : self.n_head, :seq_length_with_past, :seq_length_with_past
            ]
        return mask

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(self, enc_input_ids, enc_position_ids, enc_attn_mask,
                dec_input_ids=None, dec_position_ids=None, dec_attn_mask=None,
                retriever_input_ids=None,
                retriever_position_ids=None,
                retriever_attn_mask=None,
                enc_dec_attn_mask=None, tokentype_ids=None,
                inference_params=None,
                pooling_sequence_index=0,
                enc_hidden_states=None, output_enc_hidden=False):

        # Encoder embedding.
        if self.pre_process:
            encoder_input = self.embedding(enc_input_ids, enc_position_ids,
                                           tokentype_ids=tokentype_ids)
        else:
            encoder_input = None

        if self.use_alibi_mask:
            if self.training:
                if (
                        self.alibi_mask is None
                        or self.alibi_mask.shape[-1] != self.seq_length_with_past
                ):
                    self.alibi_mask = self.get_alibi_mask(
                        encoder_input, self.seq_length_with_past
                    )
                alibi_mask = self.alibi_mask
            else:
                alibi_mask = self.get_alibi_mask(encoder_input, self.seq_length_with_past)

            if enc_attn_mask is not None:
                if len(enc_attn_mask.shape) == 2:
                    expanded_mask = enc_attn_mask.to(alibi_mask.dtype)
                    expanded_mask = torch.tril(
                        torch.gt(expanded_mask[:, :, None] * expanded_mask[:, None, :], 0)
                    ) * torch.eq(expanded_mask[:, :, None] - expanded_mask[:, None, :], 0)
                else:
                    expanded_mask = enc_attn_mask
                bsz = encoder_input.size(0)
                src_len, tgt_len = alibi_mask.size()[-2:]
                expanded_mask = (
                    expanded_mask.unsqueeze(1)
                    .expand(bsz, 1, src_len, tgt_len)
                    .to(alibi_mask.dtype)
                )
                inverted_mask = 1.0 - expanded_mask
                inverted_mask = inverted_mask.masked_fill(
                    inverted_mask.to(torch.bool), torch.finfo(alibi_mask.dtype).min
                )
                enc_attn_mask = inverted_mask + alibi_mask.unsqueeze(0)

            else:
                enc_attn_mask = alibi_mask
        else:
            past_key_values_length = 0
            batch_size = enc_input_ids.shape[0]
            if enc_attn_mask is None:
                enc_attn_mask = torch.ones(
                    (batch_size, self.seq_length_with_past), dtype=torch.bool, device=enc_input_ids.device
                )

            enc_attn_mask = self._prepare_decoder_attention_mask(
                enc_attn_mask, (batch_size, self.seq_length), encoder_input, past_key_values_length
            )

            if enc_position_ids is None:
                device = enc_input_ids.device if enc_input_ids is not None else encoder_input.device
                enc_position_ids = torch.arange(
                    past_key_values_length, self.seq_length + past_key_values_length, dtype=torch.long, device=device
                )
                enc_position_ids = enc_position_ids.unsqueeze(0).view(-1, self.seq_length)
            else:
                enc_position_ids = enc_position_ids.view(-1, self.seq_length).long()

        # Retriever embedding.
        if self.add_retriever and self.pre_process:
            retriever_input = self.embedding(retriever_input_ids,
                                             retriever_position_ids,
                                             tokentype_ids=tokentype_ids)
        else:
            retriever_input = None

        # Rotary positional embeddings
        rotary_pos_emb = None
        if self.use_rotary_position_embeddings:
            if inference_params is not None:
                rotary_pos_emb = \
                    self.rotary_pos_emb(encoder_input, inference_params.max_sequence_length)
            else:
                rotary_pos_emb = self.rotary_pos_emb(encoder_input, self.seq_length)

        # Run encoder.
        if enc_hidden_states is None:
            if self.encoder is not None:
                encoder_output = self.encoder(
                    encoder_input,
                    enc_attn_mask,
                    retriever_input=retriever_input,
                    retriever_attn_mask=retriever_attn_mask,
                    inference_params=inference_params,
                    rotary_pos_emb=rotary_pos_emb,
                    position_ids=enc_position_ids)
            else:
                encoder_output = self.encoder_hidden_state
        else:
            encoder_output = enc_hidden_states.to(encoder_input.dtype)

        if self.post_process:
            if self.add_pooler:
                pooled_output = self.pooler(encoder_output,
                                            pooling_sequence_index)

        # output_enc_hidden refers to when we just need the encoder's
        # output. For example, it is helpful to compute
        # similarity between two sequences by average pooling
        if not self.add_decoder or output_enc_hidden:
            if self.add_pooler and self.post_process:
                return encoder_output, pooled_output
            else:
                return encoder_output

        # Decoder embedding.
        if self.pre_process:
            decoder_input = self.embedding(dec_input_ids,
                                           dec_position_ids)
        else:
            decoder_input = None

        # Run decoder.
        decoder_output = self.decoder(
            decoder_input,
            dec_attn_mask,
            encoder_output=encoder_output,
            enc_dec_attn_mask=enc_dec_attn_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            position_ids=enc_position_ids)

        if self.add_pooler and self.post_process:
            return decoder_output, encoder_output, pooled_output
        else:
            return decoder_output, encoder_output

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        if self.pre_process:
            state_dict_[self._embedding_key] \
                = self.embedding.state_dict_for_save_checkpoint(prefix=prefix,
                                                                keep_vars=keep_vars)
        if self.add_encoder:
            state_dict_[self._encoder_key] \
                = self.encoder.state_dict_for_save_checkpoint(prefix=prefix,
                                                              keep_vars=keep_vars)
        if self.post_process:
            if self.add_pooler:
                state_dict_[self._pooler_key] \
                    = self.pooler.state_dict_for_save_checkpoint(prefix=prefix,
                                                                 keep_vars=keep_vars)
            if self.untie_embeddings_and_output_weights:
                state_dict_[self._output_layer_key] \
                    = self.output_layer.state_dict(prefix=prefix, keep_vars=keep_vars)

        if self.add_decoder:
            state_dict_[self._decoder_key] \
                = self.decoder.state_dict_for_save_checkpoint(prefix=prefix,
                                                              keep_vars=keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""
        args = get_args()
        # Embedding.
        if self.pre_process:
            if self._embedding_key in state_dict:
                state_dict_ = state_dict[self._embedding_key]
            else:
                # for backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if '_embeddings' in key:
                        state_dict_[key] = state_dict[key]
            self.embedding.load_state_dict(state_dict_, strict=strict)

        # Encoder.
        if self.add_encoder:
            if self._encoder_key in state_dict:
                state_dict_ = state_dict[self._encoder_key]
            # For backward compatibility.
            elif 'transformer' in state_dict:
                state_dict_ = state_dict['transformer']
            else:
                # For backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if 'transformer.' in key:
                        state_dict_[key.split('transformer.')[1]] = state_dict[key]

            # For backward compatibility.
            state_dict_self_attention = {}
            for key in state_dict_.keys():
                if '.attention.' in key:
                    state_dict_self_attention[key.replace(".attention.",
                        ".self_attention.")] = state_dict_[key]
                else:
                    state_dict_self_attention[key] = state_dict_[key]
            state_dict_ = state_dict_self_attention

            if args.transformer_impl == "transformer_engine":
                self.encoder.load_state_dict(state_dict_, strict=False)
            else:
                self.encoder.load_state_dict(state_dict_, strict=strict)

        # Pooler.
        if self.post_process:
            if self.add_pooler:
                assert 'pooler' in state_dict, \
                    'could not find data for pooler in the checkpoint'
                self.pooler.load_state_dict(state_dict[self._pooler_key],
                                            strict=strict)
            if self.untie_embeddings_and_output_weights:
                assert 'output_layer' in state_dict, \
                    'could not find data for output_layer in the checkpoint'
                self.output_layer.load_state_dict(state_dict[self._output_layer_key],
                                                  strict=strict)
        # Decoder.
        if self.add_decoder:
            assert 'decoder' in state_dict, \
                'could not find data for pooler in the checkpoint'
            self.decoder.load_state_dict(state_dict[self._decoder_key],
                                         strict=strict)

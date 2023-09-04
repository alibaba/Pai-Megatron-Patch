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

import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron import core, get_args
from megatron.core import mpu, tensor_parallel
from megatron.model import LayerNorm
from megatron.model.enums import AttnMaskType
from megatron.model.enums import AttnType
from megatron.model.enums import LayerType
from megatron.model.enums import ModelType
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.module import MegatronModule
from megatron.model.utils import attention_mask_func
from megatron.model.utils import openai_gelu
from megatron.model.utils import erf_gelu

from .positional_embeddings import RotaryEmbedding, apply_rotary_pos_emb_index

try:
    from einops import rearrange
except ImportError:
    rearrange = None

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
except ImportError:
    flash_attn_unpadded_func = None
""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
"""


class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """
    def __init__(self, init_method, output_layer_init_method):
        super(ParallelMLP, self).__init__()
        args = get_args()

        # Project to 4h.
        self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
            args.hidden_size,
            args.ffn_hidden_size,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True)

        self.bias_gelu_fusion = args.bias_gelu_fusion
        self.activation_func = F.gelu
        if args.openai_gelu:
            self.activation_func = openai_gelu
        elif args.onnx_safe:
            self.activation_func = erf_gelu

        # Project back to h.
        self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
            args.ffn_hidden_size,
            args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True)

    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel, _ = self.dense_h_to_4h(hidden_states)

        intermediate_parallel = self.activation_func(intermediate_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, None


class CoreAttention(MegatronModule):
    def __init__(self, layer_number, attn_mask_type=AttnMaskType.padding):
        super(CoreAttention, self).__init__()
        args = get_args()
        self.fp16 = args.fp16
        self.bf16 = args.bf16
        self.position_embedding_type = args.position_embedding_type
        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.sequence_parallel = args.sequence_parallel

        projection_size = args.kv_channels * args.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = core.utils.divide(
            projection_size, world_size)
        self.hidden_size_per_attention_head = core.utils.divide(
            projection_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = core.utils.divide(
            args.num_attention_heads, world_size)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16, self.bf16, self.attn_mask_type,
            args.masked_softmax_fusion, attention_mask_func,
            self.attention_softmax_in_fp32, coeff)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(args.attention_dropout)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]

        seq_len, b, nh, hidden_size = key_layer.shape
        query_key_layer_scaling_coeff = float(self.layer_number)
        query_layer = query_layer / (math.sqrt(hidden_size) *
                                     query_key_layer_scaling_coeff)

        output_size = (query_layer.size(1), query_layer.size(2),
                       query_layer.size(0), key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)

        matmul_result = torch.zeros(
            1,
            1,
            1,
            dtype=query_layer.dtype,
            device=query_layer.device,
        )

        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=1.0,
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        """
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)
        """
        if not (attention_mask == 0).all():
            # if auto-regressive, skip
            attention_scores.masked_fill_(attention_mask, -10000.0)
        dtype = attention_scores.dtype
        attention_scores = attention_scores.float()
        attention_scores = attention_scores * query_key_layer_scaling_coeff

        attention_probs = F.softmax(attention_scores, dim=-1)

        attention_probs = attention_probs.type(dtype)

        # This is actually dropping out entire tokens to attend to,
        # which might seem a bit unusual, but is taken from
        # the original Transformer paper.

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1), value_layer.size(2),
                       query_layer.size(0), value_layer.size(3))

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape =\
            context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class FlashSelfAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self,
                 causal=False,
                 softmax_scale=None,
                 attention_dropout=0.0,
                 device=None,
                 dtype=None):
        super().__init__()
        assert flash_attn_unpadded_func is not None, (
            'Please install FlashAttention first, '
            'e.g., with pip install flash-attn')
        assert rearrange is not None, 'Please' \
                                      ' install einops first,' \
                                      ' e.g., with pip install einops'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, q, k, v):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the
             query, key, and value. (B, S, H, D)
        """
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda
        batch_size, seqlen = q.shape[0], q.shape[1]
        q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
        max_s = seqlen
        cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen,
                                  step=seqlen,
                                  dtype=torch.int32,
                                  device=q.device)
        output = flash_attn_unpadded_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_s,
            max_s,
            self.dropout_p if self.training else 0.0,
            softmax_scale=self.softmax_scale,
            causal=self.causal)
        output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        return output


class ParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """
    def __init__(self,
                 init_method,
                 output_layer_init_method,
                 layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.padding):
        super(ParallelAttention, self).__init__()
        args = get_args()
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.params_dtype = args.params_dtype
        self.sequence_parallel = args.sequence_parallel
        self.position_embedding_type = args.position_embedding_type
        self.bf16 = args.bf16
        self.use_flash_attn = args.use_flash_attn
        self.position_encoding_2d = args.position_encoding_2d
        if self.use_flash_attn:
            if flash_attn_unpadded_func is None:
                raise ImportError(
                    'FlashAttention is not installed, please install with '
                    'pip install flash-attn')
            assert attention_type == AttnType.self_attn, (
                'FlashAttention code path only supports '
                'self-attention for now')
            assert self.attn_mask_type == AttnMaskType.causal, (
                'FlashAttention code path only '
                'supports causal mask for now')
            if rearrange is None:
                raise ImportError('einops is not installed,'
                                  ' please install with pip install einops')

        projection_size = args.kv_channels * args.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = core.utils.divide(
            projection_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = core.utils.divide(
            args.num_attention_heads, world_size)

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
            self.query_key_value = tensor_parallel.ColumnParallelLinear(
                args.hidden_size,
                3 * projection_size,
                gather_output=False,
                init_method=init_method)
        else:
            assert attention_type == AttnType.cross_attn
            self.query = tensor_parallel.ColumnParallelLinear(
                args.hidden_size,
                projection_size,
                gather_output=False,
                init_method=init_method)

            self.key_value = tensor_parallel.ColumnParallelLinear(
                args.hidden_size,
                2 * projection_size,
                gather_output=False,
                init_method=init_method)

        self.core_attention = CoreAttention(self.layer_number,
                                            self.attn_mask_type)
        self.checkpoint_core_attention = \
            args.recompute_granularity == 'selective'

        if self.use_flash_attn:
            self.core_attention_flash = FlashSelfAttention(
                causal=True, attention_dropout=args.attention_dropout)
        # Output.
        self.dense = tensor_parallel.RowParallelLinear(
            projection_size,
            args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True)

        if self.position_embedding_type == 'rotary':
            dim = self.hidden_size_per_attention_head
            self.rotary_emb = RotaryEmbedding(dim / 2)
        else:
            self.rotary_emb = None

    def _checkpointed_attention_forward(self, query_layer, key_layer,
                                        value_layer, attention_mask):
        """Forward method with activation checkpointing."""
        def custom_forward(*inputs):
            query_layer = inputs[0]
            key_layer = inputs[1]
            value_layer = inputs[2]
            attention_mask = inputs[3]
            output_ = self.core_attention(query_layer, key_layer, value_layer,
                                          attention_mask)
            return output_

        hidden_states = tensor_parallel.checkpoint(custom_forward, False,
                                                   query_layer, key_layer,
                                                   value_layer, attention_mask)

        return hidden_states

    def _allocate_memory(self, inference_max_sequence_len, batch_size):
        return torch.empty(inference_max_sequence_len,
                           batch_size,
                           self.num_attention_heads_per_partition,
                           self.hidden_size_per_attention_head,
                           dtype=self.params_dtype,
                           device=torch.cuda.current_device())

    def forward(self,
                hidden_states,
                position_ids,
                attention_mask,
                encoder_output=None,
                inference_params=None):
        # hidden_states: [sq, b, h]
        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer, _ = self.query_key_value(hidden_states)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape =\
            mixed_x_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        # torch.Size([128, 1, 32, 128])
        (query_layer, key_layer,
         value_layer) = tensor_parallel.split_tensor_along_last_dim(
             mixed_x_layer, 3)
        if self.position_encoding_2d:
            q1, q2 = query_layer.chunk(2, dim=(query_layer.ndim - 1))
            k1, k2 = key_layer.chunk(2, dim=(key_layer.ndim - 1))
            cos, sin = self.rotary_emb(q1, seq_len=position_ids.max() + 1)
            position_ids, block_position_ids =\
                position_ids[:, 0, :].transpose(0, 1).contiguous(), \
                position_ids[:, 1, :].transpose(0, 1).contiguous()
            # q1: torch.Size([128, 4, 32, 64])
            # position_ids :torch.Size([128, 4])
            q1, k1 = apply_rotary_pos_emb_index(q1, k1, cos, sin, position_ids)
            q2, k2 = apply_rotary_pos_emb_index(q2, k2, cos, sin,
                                                block_position_ids)
            query_layer = torch.concat([q1, q2], dim=(q1.ndim - 1))
            key_layer = torch.concat([k1, k2], dim=(k1.ndim - 1))
        else:
            position_ids = position_ids.transpose(0, 1)
            cos, sin = self.rotary_emb(value_layer,
                                       seq_len=position_ids.max() + 1)
            # [seq_len, batch, num_attention_heads,
            # hidden_size_per_attention_head]
            query_layer, key_layer = apply_rotary_pos_emb_index(
                query_layer, key_layer, cos, sin, position_ids)

        # ==================================
        # core attention computation
        # ==================================

        if not self.use_flash_attn:
            if self.checkpoint_core_attention:
                context_layer = self._checkpointed_attention_forward(
                    query_layer, key_layer, value_layer, attention_mask)
            else:
                context_layer = self.core_attention(query_layer, key_layer,
                                                    value_layer,
                                                    attention_mask)
        else:
            q, k, v = [
                rearrange(x, 's b ... -> b s ...').contiguous()
                for x in (query_layer, key_layer, value_layer)
            ]
            if not self.sequence_parallel:
                with tensor_parallel.get_cuda_rng_tracker().fork():
                    context_layer = self.core_attention_flash(q, k, v)
            else:
                context_layer = self.core_attention_flash(q, k, v)
            context_layer = rearrange(context_layer,
                                      'b s h d -> s b (h d)').contiguous()

        # =================
        # Output. [sq, b, h]
        # =================
        output, bias = self.dense(context_layer)

        return output, bias


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)

    return _bias_dropout_add


class ParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """
    def __init__(self,
                 init_method,
                 output_layer_init_method,
                 layer_number,
                 layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding):

        args = get_args()

        super(ParallelTransformerLayer, self).__init__()
        self.layer_number = layer_number
        self.layer_type = layer_type
        self.num_layers = args.num_layers
        self.apply_residual_connection_post_layernorm \
            = args.apply_residual_connection_post_layernorm

        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            no_persist_layer_norm=args.no_persist_layer_norm,
            sequence_parallel=args.sequence_parallel)

        # Self attention.
        self.self_attention = ParallelAttention(
            init_method,
            output_layer_init_method,
            layer_number,
            attention_type=AttnType.self_attn,
            attn_mask_type=self_attn_mask_type)
        self.hidden_dropout = args.hidden_dropout
        self.bias_dropout_fusion = args.bias_dropout_fusion

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            no_persist_layer_norm=args.no_persist_layer_norm,
            sequence_parallel=args.sequence_parallel)

        if self.layer_type == LayerType.decoder:
            self.inter_attention = ParallelAttention(
                init_method,
                output_layer_init_method,
                layer_number,
                attention_type=AttnType.cross_attn)
            # Layernorm on the attention output.
            self.post_inter_attention_layernorm = LayerNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon,
                no_persist_layer_norm=args.no_persist_layer_norm,
                sequence_parallel=args.sequence_parallel)

        self.sequence_parallel = args.sequence_parallel
        self.mlp = ParallelMLP(init_method, output_layer_init_method)

        # Set bias+dropout+add fusion grad_enable execution handler.
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1
                                          and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = nullcontext \
            if use_nvfuser else torch.enable_grad

    def forward(self,
                hidden_states,
                position_ids,
                attention_mask,
                encoder_output=None,
                enc_dec_attn_mask=None,
                inference_params=None):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, attention_bias = \
            self.self_attention(
                layernorm_output,
                position_ids,
                attention_mask,
                inference_params=inference_params)

        alpha = (2 * self.num_layers)**0.5
        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output * alpha
        else:
            residual = hidden_states

        layernorm_input = attention_output + residual

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        args = get_args()

        if args.recompute_granularity == 'selective':

            def remove_bias_forward(layernorm_output):
                mlp_output, mlp_bias = self.mlp(layernorm_output)
                return mlp_output

            mlp_output = tensor_parallel.checkpoint(remove_bias_forward, False,
                                                    layernorm_output)
            mlp_bias = self.mlp.dense_4h_to_h.bias \
                if self.mlp.dense_4h_to_h.skip_bias_add else None
        else:
            mlp_output, mlp_bias = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output * alpha
        else:
            residual = layernorm_input

        output = mlp_output + residual

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = core.utils.make_viewless_tensor(
            inp=output, requires_grad=output.requires_grad, keep_graph=True)

        return output


def _get_num_layers(args, is_encoder_and_decoder_model, is_decoder=False):
    """Compute the number of transformer
     layers resident on the current rank."""
    if mpu.get_pipeline_model_parallel_world_size() > 1:
        if is_encoder_and_decoder_model:
            assert args.pipeline_model_parallel_split_rank is not None

            # When a standalone embedding stage is used, a rank is taken from
            # the encoder's ranks, to be used for the encoder's embedding
            # layer. This way, the rank referenced by the 'split rank' remains
            # the same whether or not a standalone embedding stage is used.
            num_ranks_in_encoder = (args.pipeline_model_parallel_split_rank -
                                    1 if args.standalone_embedding_stage else
                                    args.pipeline_model_parallel_split_rank)
            num_ranks_in_decoder =\
                args.transformer_pipeline_model_parallel_size -\
                num_ranks_in_encoder
            assert args.encoder_num_layers % num_ranks_in_encoder == 0
            assert args.decoder_num_layers % num_ranks_in_decoder == 0
            if mpu.is_pipeline_stage_before_split():
                num_layers = (0 if args.standalone_embedding_stage
                              and mpu.get_pipeline_model_parallel_rank() == 0
                              else args.encoder_num_layers //
                              num_ranks_in_encoder)
            else:
                num_layers = args.decoder_num_layers // num_ranks_in_decoder
        else:
            assert args.num_layers == args.encoder_num_layers
            assert args.num_layers %\
                   args.transformer_pipeline_model_parallel_size == 0

            num_layers = (0 if args.standalone_embedding_stage
                          and mpu.get_pipeline_model_parallel_rank() == 0 else
                          args.num_layers //
                          args.transformer_pipeline_model_parallel_size)
    else:
        if not is_decoder:
            num_layers = args.encoder_num_layers
        else:
            num_layers = args.decoder_num_layers
    return num_layers


class ParallelTransformer(MegatronModule):
    """Transformer class."""
    def __init__(self,
                 init_method,
                 output_layer_init_method,
                 layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 post_layer_norm=True,
                 pre_process=True,
                 post_process=True,
                 drop_path_rate=0.0,
                 position_encoding_2d=False):
        super(ParallelTransformer, self).__init__()
        args = get_args()

        self.layer_type = layer_type
        self.model_type = args.model_type
        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None
        self.position_encoding_2d = position_encoding_2d

        # Store activation checkpoiting flag.
        self.recompute_granularity = args.recompute_granularity
        self.recompute_method = args.recompute_method
        self.recompute_num_layers = args.recompute_num_layers
        self.distribute_saved_activations = \
            args.distribute_saved_activations and not args.sequence_parallel

        self.sequence_parallel = args.sequence_parallel

        # Number of layers.
        self.num_layers = _get_num_layers(
            args, args.model_type == ModelType.encoder_and_decoder)

        # Transformer layers.
        def build_layer(layer_number):
            return ParallelTransformerLayer(
                init_method,
                output_layer_init_method,
                layer_number,
                layer_type=layer_type,
                self_attn_mask_type=self_attn_mask_type)

        if args.virtual_pipeline_model_parallel_size is not None:
            assert args.num_layers % \
                   args.virtual_pipeline_model_parallel_size == 0
            assert args.model_type != ModelType.encoder_and_decoder
            # Number of layers in each model chunk
            # is the number of layers in the stage,
            # divided by the number of model chunks in a stage.
            self.num_layers = \
                self.num_layers // args.virtual_pipeline_model_parallel_size
            # With 8 layers, 2 stages, and 4 model chunks,
            # we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0]  [2]  [4]  [6]
            # Stage 1: [1]  [3]  [5]  [7]
            # With 8 layers, 2 stages, and 2 virtual stages,
            # we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]  [4, 5]
            # Stage 1: [2, 3]  [6, 7]
            offset =\
                mpu.get_virtual_pipeline_model_parallel_rank() * (
                        args.num_layers //
                        args.virtual_pipeline_model_parallel_size) + (
                        mpu.get_pipeline_model_parallel_rank() *
                        self.num_layers)
        else:
            # Each stage gets a contiguous set of layers.
            if args.model_type == ModelType.encoder_and_decoder and \
                    mpu.get_pipeline_model_parallel_world_size() > 1:
                pipeline_rank = mpu.get_pipeline_model_parallel_rank()
                if layer_type == LayerType.encoder:
                    offset = pipeline_rank * self.num_layers
                else:
                    num_ranks_in_enc = args.pipeline_model_parallel_split_rank
                    offset = (pipeline_rank -
                              num_ranks_in_enc) * self.num_layers
            else:
                offset = mpu.get_pipeline_model_parallel_rank(
                ) * self.num_layers

        self.layers = torch.nn.ModuleList(
            [build_layer(i + 1 + offset) for i in range(self.num_layers)])

        if self.post_process and self.post_layer_norm:
            # Final layer norm before output.
            self.final_layernorm = LayerNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon,
                no_persist_layer_norm=args.no_persist_layer_norm,
                sequence_parallel=args.sequence_parallel)

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def _checkpointed_forward(self, hidden_states, position_ids,
                              attention_mask, encoder_output,
                              enc_dec_attn_mask):
        """Forward method with activation checkpointing."""
        def custom(start, end):
            def custom_forward(*inputs):
                x_ = inputs[0]
                position_ids = inputs[1]
                attention_mask = inputs[2]
                encoder_output = inputs[3]
                enc_dec_attn_mask = inputs[4]
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(x_, position_ids, attention_mask,
                               encoder_output, enc_dec_attn_mask)

                return x_

            return custom_forward

        if self.recompute_method == 'uniform':
            # Uniformly divide the total
            # number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            layer = 0
            while layer < self.num_layers:
                hidden_states = tensor_parallel.checkpoint(
                    custom(layer, layer + self.recompute_num_layers),
                    self.distribute_saved_activations, hidden_states,
                    position_ids, attention_mask, encoder_output,
                    enc_dec_attn_mask)
                layer += self.recompute_num_layers

        elif self.recompute_method == 'block':
            # Checkpoint the input activation
            # of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory
            # removing redundant re-computation.
            for layer in range(self.num_layers):
                if layer < self.recompute_num_layers:
                    hidden_states = tensor_parallel.checkpoint(
                        custom(layer,
                               layer + 1), self.distribute_saved_activations,
                        hidden_states, position_ids, attention_mask,
                        encoder_output, enc_dec_attn_mask)

                else:
                    hidden_states = custom(layer, layer + 1)(hidden_states,
                                                             position_ids,
                                                             attention_mask,
                                                             encoder_output,
                                                             enc_dec_attn_mask)

        else:
            raise ValueError('Invalid activation recompute method.')

        return hidden_states

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(self,
                hidden_states,
                position_ids,
                attention_mask,
                encoder_output=None,
                enc_dec_attn_mask=None):
        # hidden_states: [s, b, h]
        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = core.utils.make_viewless_tensor(
            hidden_states,
            requires_grad=True,
            keep_graph=True,
        )

        if self.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        with rng_context:
            # Forward pass.
            if self.recompute_granularity == 'full':
                hidden_states = self._checkpointed_forward(
                    hidden_states, position_ids, attention_mask,
                    encoder_output, enc_dec_attn_mask)
            else:
                for index in range(self.num_layers):
                    layer = self._get_layer(index)
                    hidden_states = layer(hidden_states,
                                          position_ids,
                                          attention_mask,
                                          encoder_output=encoder_output,
                                          enc_dec_attn_mask=enc_dec_attn_mask)

        # Final layer norm.
        if self.post_process and self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states

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

from megatron import core, get_timers, get_args
from megatron.core import mpu, tensor_parallel
from megatron.utils import print_rank_0
from megatron.model.module import MegatronModule
from .enums import AttnMaskType, ModelType, LayerType, AttnType, PositionEmbeddingType
from megatron.model.fused_layer_norm import MixedFusedLayerNorm as LayerNorm
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.utils import attention_mask_func
from megatron.model.utils import openai_gelu
from megatron.model.utils import erf_gelu
from megatron.model.utils import get_linear_layer


from .glu_activations import GLU_ACTIVATIONS

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

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

class DropPath(MegatronModule):
    """Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_state):
        if self.drop_prob == 0. or not self.training:
            return hidden_state
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        shape = (hidden_state.shape[0],) + (1,) * (hidden_state.ndim - 1)
        random_tensor = keep_prob + \
            torch.rand(shape, dtype=hidden_state.dtype, device=hidden_state.device)
        random_tensor.floor_()  # binarize
        output = hidden_state.div(keep_prob) * random_tensor
        return output


class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(self, init_method, output_layer_init_method):
        super(ParallelMLP, self).__init__()
        args = get_args()

        # Project to ffn_hidden_size
        self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
            args.hidden_size,
            # GLU is a special activation that divides the dimension by a factor 2.
            2 * args.ffn_hidden_size if args.glu_activation else args.ffn_hidden_size,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True)

        self.bias_gelu_fusion = args.bias_gelu_fusion
        self.activation_func = F.gelu
        if args.glu_activation:
            self.activation_func = GLU_ACTIVATIONS[args.glu_activation]
        elif args.openai_gelu:
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
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if self.bias_gelu_fusion:
             intermediate_parallel = \
                     bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            intermediate_parallel = \
                self.activation_func(intermediate_parallel + bias_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias

class SwitchMLP(MegatronModule):
    """
    Routes input to one of N MLP "experts"
    """
    def __init__(self, init_method, output_layer_init_method):
        super(SwitchMLP, self).__init__()
        args = get_args()
        self.router = torch.nn.Linear(args.hidden_size, args.num_experts)
        self.experts = torch.nn.ModuleList()
        for i in range(args.num_experts):
            self.experts.append(ParallelMLP(init_method, output_layer_init_method))

    def forward(self, hidden_states):
        # hidden_states: [s, b, h]
        s = hidden_states.size(0)
        b = hidden_states.size(1)
        h = hidden_states.size(2)
        route = self.router(hidden_states)
        route = torch.nn.functional.softmax(route, dim=2)
        max_prob, max_ind = torch.max(route, dim=2)
        max_prob = torch.unsqueeze(max_prob, 2) # [s b 1]

        # TODO (rprenger) TODO this could be made easier to read
        # Converting [s, b, h] to [s*b, h].
        # Each vector could be routed differently
        hidden_states = hidden_states.view(-1, hidden_states.size(2)) # [s*b h]
        max_prob = max_prob.view(-1, max_prob.size(2)) # [s*b 1]
        max_ind = max_ind.view(-1) # [s*b]

        output_total = torch.empty_like(hidden_states)
        output_bias_total = torch.empty_like(hidden_states)
        #TODO (rprenger) This does each expert in serial, but it could be parallelized

        for expert_num, expert in enumerate(self.experts):
            local_indices = (max_ind == expert_num).nonzero()
            hidden = hidden_states[local_indices,:]
            output, output_bias = expert(hidden)
            output_bias = output_bias.expand_as(output)
            output_total[local_indices,:] = output
            output_bias_total[local_indices,:] = output_bias

        output_total = output_total*max_prob
        output_bias_total = output_bias_total*max_prob
        output_total = output_total.view(s, b, h)
        output_bias_total = output_bias_total.view(s, b, h)

        return output_total, output_bias_total


class CoreAttention(MegatronModule):

    def __init__(self, layer_number,
                 attn_mask_type=AttnMaskType.padding):
        super(CoreAttention, self).__init__()
        args = get_args()
        self.fp16 = args.fp16
        self.bf16 = args.bf16

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
        self.hidden_size_per_partition = core.utils.divide(projection_size,
                                                    world_size)
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
            self.fp16, self.bf16,
            self.attn_mask_type,
            args.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(args.attention_dropout)

    def forward(self, query_layer, key_layer,
                value_layer, attention_mask, alibi):

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================
        np = query_layer.size(2)

        # [b, np, sq, sk]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)

        if alibi is None:
            # preallocting input tensor: [b * np, sq, sk]
            matmul_input_buffer = mpu.get_global_memory_buffer().get_tensor(
                (output_size[0]*output_size[1], output_size[2], output_size[3]),
                query_layer.dtype, "mpu")
        else:
            # alibi: (batch_size * num_attention_heads, 1, max_seq_len)
            matmul_input_buffer = alibi[:output_size[0]*output_size[1], :, :output_size[3]]

        # Raw attention scores. [b * np, sq, sk]
        if alibi is None:
            matmul_result = torch.baddbmm(
                matmul_input_buffer,
                query_layer.transpose(0, 1),   # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0, alpha=(1.0/self.norm_factor))
        else:
            if not hasattr(self, "logged_alibi"):
                print("Using Alibi.")
                self.logged_alibi = True

            if self.apply_query_key_layer_scaling:
                beta = 1.0 / self.layer_number
            else:
                beta = 1.0

            matmul_result = torch.baddbmm(
                matmul_input_buffer,
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=beta, alpha=(1.0 / self.norm_factor))

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1),
                       np,
                       query_layer.size(0),
                       value_layer.size(3))

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
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class MultiQueryCoreAttention(CoreAttention):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, query_layer, key_layer, value_layer, attention_mask, alibi):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================
        sq = query_layer.size(0)
        bs = query_layer.size(1)
        np = query_layer.size(2)

        sk = key_layer.size(0)
        # Only one head for key and values
        assert key_layer.size(2) == 1 and value_layer.size(2) == 1

        # [b, np, sq, sk]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        # [sq, b, np, hn] -> [b, np * sq, hn]
        query_layer = query_layer.permute([1, 2, 0, 3]).reshape(bs, np * sq, -1)
        # [sk, b, 1, hn] -> [b, hn, sk]
        key_layer = key_layer.squeeze(2).permute(1, 2, 0)
        # [sk, b, 1, hn] -> [sk, b * np, hn]
        # key_layer = key_layer.expand(output_size[3], output_size[0], np, -1)
        # key_layer = key_layer.reshape(output_size[3], output_size[0] * np, -1)

        if alibi is None:
            # preallocting input tensor: [b, np * sq, sk]
            matmul_input_buffer = mpu.get_global_memory_buffer().get_tensor(
                (bs, np * sq, sk),
                query_layer.dtype, "mpu")
        else:
            # alibi: (batch_size * num_attention_heads, 1, max_seq_len)
            # TODO: ideally, alibi would have the shape: (1, num_heads * sq, sk)
            matmul_input_buffer = alibi[:bs * np, :, :sk].view(bs, np, sk)
            matmul_input_buffer = matmul_input_buffer.repeat(1, sq, 1)  # [b, np * sq, sk]

        if alibi is None:
            # Raw attention scores. [b, np * sq, sk]
            matmul_result = torch.baddbmm(
                matmul_input_buffer,
                query_layer,   # [b, np * sq, hn]
                key_layer,  # [b, hn, sk]
                beta=0.0, alpha=(1.0/self.norm_factor))
        else:
            if not hasattr(self, "logged_alibi"):
                print("Using Alibi.")
                self.logged_alibi = True

            if self.apply_query_key_layer_scaling:
                beta = 1.0 / self.layer_number
            else:
                beta = 1.0

            matmul_result = torch.baddbmm(
                matmul_input_buffer,
                query_layer,
                key_layer,
                beta=beta, alpha=(1.0 / self.norm_factor))

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(bs, np, sq, sk)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1),
                       np,
                       query_layer.size(0),
                       value_layer.size(3))

        # [sk, b, 1, hn] -> [b, sk, hn]
        value_layer = value_layer.squeeze(2).transpose(0, 1)

        # change view [b, np * sq, sk]
        attention_probs = attention_probs.view(bs, np * sq, -1)

        # matmul: [b, np * sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer)

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(bs, np, sq, -1)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
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
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 device=None, dtype=None):
        super().__init__()
        assert flash_attn_unpadded_func is not None, ('Please install FlashAttention first, '
                                                      'e.g., with pip install flash-attn')
        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, q, k, v):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda
        batch_size, seqlen = q.shape[0], q.shape[1]
        q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
        max_s = seqlen
        cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                  device=q.device)
        output = flash_attn_unpadded_func(
            q, k, v, cu_seqlens, cu_seqlens, max_s, max_s,
            self.dropout_p if self.training else 0.0,
            softmax_scale=self.softmax_scale, causal=self.causal
        )
        output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        return output


class ParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, init_method,
                 output_layer_init_method, layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.padding):
        super(ParallelAttention, self).__init__()
        args = get_args()
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.params_dtype = args.params_dtype
        self.attention_head_type = args.attention_head_type
        self.sequence_parallel = args.sequence_parallel

        self.use_flash_attn = args.use_flash_attn

        projection_size = args.kv_channels * args.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = core.utils.divide(
            projection_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = core.utils.divide(
            args.num_attention_heads, world_size)

        # Strided linear layer.
        if attention_type == AttnType.self_attn and self.attention_head_type == 'multihead':
            self.query_key_value = tensor_parallel.ColumnParallelLinear(
                args.hidden_size,
                3 * projection_size,
                gather_output=False,
                init_method=init_method)
        elif attention_type == AttnType.self_attn and self.attention_head_type == 'multiquery':
            # TODO: Find a way to merge the query and key-value computations?
            self.query = tensor_parallel.ColumnParallelLinear(
                args.hidden_size,
                projection_size,
                gather_output=False,
                init_method=init_method)
            # In MultiQuery attention, keys and values are shared across heads
            # Use args.kv_channels instead of projection_size
            # No `.fork()` so the rng tracker is shared across tensor-parallel processes.
            # with tensor_parallel.get_cuda_rng_tracker():
            self.key_value = get_linear_layer(
                args.hidden_size,
                2 * args.kv_channels,
                init_method=init_method)
        elif attention_type == AttnType.cross_attn and self.attention_head_type == 'multihead':
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
        elif attention_type == AttnType.cross_attn and self.attention_head_type == 'multiquery':
            raise NotImplementedError("Multiquery attention not implemented for cross-attention.")
        else:
            raise ValueError(f"Invalid attention arguments: {attention_type}, {self.attention_head_type}")

        if self.attention_head_type == 'multihead':
            self.core_attention = CoreAttention(self.layer_number,
                                                self.attn_mask_type)
        else:
            self.core_attention = MultiQueryCoreAttention(self.layer_number, self.attn_mask_type)
        self.checkpoint_core_attention = args.recompute_granularity == 'selective'

        if self.use_flash_attn:
            if flash_attn_unpadded_func is None:
                raise ImportError('FlashAttention is not installed, please install with '
                                  'pip install flash-attn')
            assert attention_type == AttnType.self_attn, ('FlashAttention code path only supports '
                                                          'self-attention for now')
            assert self.attn_mask_type == AttnMaskType.causal, ('FlashAttention code path only '
                                                                'supports causal mask for now')
            assert args.position_embedding_type != PositionEmbeddingType.alibi, \
                ('FlashAttention does not support alibi positional embeddings yet')
            if rearrange is None:
                raise ImportError('einops is not installed, please install with pip install einops')

            if self.checkpoint_core_attention:
                print_rank_0("  Warning, using selective recomputation with flash-attn: this is already handled in the "
                             "flash-attn library and has no effect.")
            self.core_attention_flash = FlashSelfAttention(
                causal=True, attention_dropout=args.attention_dropout
            )

        # Output.
        self.dense = tensor_parallel.RowParallelLinear(
            projection_size,
            args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True)

    def _checkpointed_attention_forward(self, query_layer, key_layer,
                                        value_layer, attention_mask, alibi):
        """Forward method with activation checkpointing."""
        def custom_forward(*inputs):
            query_layer = inputs[0]
            key_layer = inputs[1]
            value_layer = inputs[2]
            attention_mask = inputs[3]
            alibi = inputs[4]
            output_ = self.core_attention(query_layer, key_layer,
                                          value_layer, attention_mask, alibi)
            return output_

        hidden_states = tensor_parallel.checkpoint(
            custom_forward,
            False, query_layer, key_layer, value_layer, attention_mask, alibi)

        return hidden_states

    def _allocate_memory(self, inference_max_sequence_len, batch_size):
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            self.num_attention_heads_per_partition if self.attention_head_type == "multihead" else 1,
            self.hidden_size_per_attention_head,
            dtype=self.params_dtype,
            device=torch.cuda.current_device())


    def forward(self, hidden_states, attention_mask,
                encoder_output=None, inference_params=None, alibi=None):
        # hidden_states: [sq, b, h]
        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        if inference_params:
            if self.layer_number not in inference_params.key_value_memory_dict:
                inf_max_seq_len = inference_params.max_sequence_len
                inf_max_batch_size = inference_params.max_batch_size
                inference_key_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size)
                inference_value_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size)
                inference_params.key_value_memory_dict[self.layer_number] = (
                    inference_key_memory, inference_value_memory)
            else:
                inference_key_memory, inference_value_memory = \
                    inference_params.key_value_memory_dict[self.layer_number]

        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == AttnType.self_attn and self.attention_head_type == 'multihead':
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer,
             key_layer,
             value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_x_layer, 3)
        elif self.attention_type == AttnType.self_attn and self.attention_head_type == 'multiquery':
            kv_input=hidden_states
            # Attention heads [sq, b, h] --> [sq, b, (2 * hn)]
            mixed_kv_layer = self.key_value(kv_input)

            # Reduce the KV gradients in the tensor-parallel direction.
            # This is different from multi-head attention which reduces the KV input,
            # because the sum over attn heads happens in the attn weight gradient instead of the KV layer:
            #   A [b, n * sq, sk] = Q [b, n * sq, hn] x K^T [b, hn, sk]
            #   G_K [b, sk, hn] = G_A [b, sk, n * sq] x Q [b, n * sq, hn]
            #                   = sum_p (G_Ap [b, sk, np * sq] x Q_p [b, np * sq, hn])
            if get_args().sequence_parallel:
                # We switch to the tensor parallel regime here instead of at the KV input
                # so that the KV layer is done in parallel instead of just duplicated.
                mixed_kv_layer = tensor_parallel.gather_from_sequence_parallel_region(mixed_kv_layer, tensor_parallel_output_grad=True)
            else:
                mixed_kv_layer = tensor_parallel.copy_to_tensor_model_parallel_region(mixed_kv_layer)

            # [sq, b, (2 * hn)] --> [sq, b, np (expanded), 2 * hn]
            # new_tensor_shape = mixed_kv_layer.size()[:-1] + \
            #     (self.num_attention_heads_per_partition,
            #      2 * self.hidden_size_per_attention_head)
            # mixed_kv_layer = mixed_kv_layer.unsqueeze(2).expand(*new_tensor_shape)

            # [sq, b, (2 * hn)] --> [sq, b, 1, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + \
                (1,
                 2 * self.hidden_size_per_attention_head)
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sq, b, np, 2 * hn] --> 2 [sq, b, np, hn]
            (key_layer,
             value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_kv_layer, 2)

            # Attention head [sq, b, h] --> [sq, b, np * hn]
            query_layer, _ = self.query(hidden_states)
            # [sq, b, np * hn] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 self.hidden_size_per_attention_head)
            query_layer = query_layer.view(*new_tensor_shape)

            # [sq, b, np, hn] -> [b, np * sq, hn]
        else:
            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer, _ = self.key_value(encoder_output)

            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 2 * self.hidden_size_per_attention_head)
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            (key_layer,
             value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_kv_layer, 2)

            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer, _ = self.query(hidden_states)
            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 self.hidden_size_per_attention_head)
            query_layer = query_layer.view(*new_tensor_shape)

        # ==================================
        # Adjust key and value for inference
        # ==================================


        if inference_params:
            batch_start = inference_params.batch_size_offset
            batch_end = batch_start + key_layer.size(1)
            assert batch_end <= inference_key_memory.size(1)
            sequence_start = inference_params.sequence_len_offset
            sequence_end = sequence_start + key_layer.size(0)
            assert sequence_end <= inference_key_memory.size(0)
            # Copy key and values.
            inference_key_memory[sequence_start:sequence_end,
                                 batch_start:batch_end, ...] = key_layer
            inference_value_memory[sequence_start:sequence_end,
                                   batch_start:batch_end, ...] = value_layer
            key_layer = inference_key_memory[
                :sequence_end, batch_start:batch_end, ...]
            value_layer = inference_value_memory[
                :sequence_end, batch_start:batch_end, ...]

        # ==================================
        # core attention computation
        # ==================================
        if self.use_flash_attn:
            if self.attention_head_type == "multiquery":
                sq, b, np, hn = query_layer.size()
                # Expand kv to be compatible with flash-attn implementation
                # [sq, b, 1, hn] -> [sq, b, np, hn]
                key_layer = key_layer.expand((sq, b, np, hn))
                value_layer = value_layer.expand((sq, b, np, hn))
            q, k, v = [rearrange(x, 's b ... -> b s ...').contiguous()
                       for x in (query_layer, key_layer, value_layer)]
            if self.sequence_parallel:
                context_layer = self.core_attention_flash(q, k, v)
            else:
                with tensor_parallel.get_cuda_rng_tracker().fork():
                    context_layer = self.core_attention_flash(q, k, v)
            context_layer = rearrange(context_layer, 'b s h d -> s b (h d)').contiguous()

        else:
            if self.checkpoint_core_attention:
                context_layer = self._checkpointed_attention_forward(
                    query_layer, key_layer, value_layer, attention_mask, alibi)
            else:
                context_layer = self.core_attention(
                    query_layer, key_layer, value_layer, attention_mask, alibi)


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


@torch.jit.script
def bias_dropout_add_fused_train(x: torch.Tensor,
                                 bias: torch.Tensor,
                                 residual: torch.Tensor,
                                 prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(x: torch.Tensor,
                                     bias: torch.Tensor,
                                     residual: torch.Tensor,
                                     prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, False)


class ParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, init_method, output_layer_init_method,
                 layer_number, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 drop_path_rate=0.):
        args = get_args()

        super(ParallelTransformerLayer, self).__init__()
        self.layer_number = layer_number
        self.layer_type = layer_type

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
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else None

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

        # MLP
        if args.num_experts is not None:
            self.mlp = SwitchMLP(init_method, output_layer_init_method)
        else:
            self.mlp = ParallelMLP(init_method, output_layer_init_method)

                    # Set bias+dropout+add fusion grad_enable execution handler.
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = \
                nullcontext if use_nvfuser else torch.enable_grad

        # Alibi
        if args.position_embedding_type == PositionEmbeddingType.alibi:
            self.alibi = self._build_alibi_tensor(args.seq_length, args.num_attention_heads, args.micro_batch_size).to(torch.cuda.current_device())
            if args.params_dtype == torch.float16:
                self.alibi = self.alibi.to(torch.float16)
            elif args.params_dtype == torch.bfloat16:
                self.alibi = self.alibi.to(torch.bfloat16)
        else:
            self.alibi = None

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                inference_params=None):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, attention_bias = \
            self.self_attention(
                layernorm_output,
                attention_mask,
                inference_params=inference_params,
                alibi=self.alibi)

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        if self.drop_path is None:
            # jit scripting for a nn.module (with dropout) is not
            # trigerring the fusion kernel. For now, we use two
            # different nn.functional routines to account for varying
            # dropout semantics during training and inference phases.
            if self.bias_dropout_fusion:
                if self.training:
                    bias_dropout_add_func = bias_dropout_add_fused_train
                else:
                    bias_dropout_add_func = bias_dropout_add_fused_inference
            else:
                bias_dropout_add_func = get_bias_dropout_add(self.training)

            with self.bias_dropout_add_exec_handler():
                layernorm_input = bias_dropout_add_func(
                    attention_output,
                    attention_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)
        else:
            out = torch.nn.functional.dropout(attention_output + attention_bias,
                                              p=self.hidden_dropout,
                                              training=self.training)
            layernorm_input = residual + self.drop_path(out)

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        if self.layer_type == LayerType.decoder:
            attention_output, attention_bias = \
                self.inter_attention(layernorm_output,
                                     enc_dec_attn_mask,
                                     encoder_output=encoder_output)
            # residual connection
            if self.apply_residual_connection_post_layernorm:
                residual = layernorm_output
            else:
                residual = layernorm_input

            with self.bias_dropout_add_exec_handler():
                layernorm_input = bias_dropout_add_func(
                    attention_output,
                    attention_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)

            # Layer norm post the decoder attention
            layernorm_output = self.post_inter_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output, mlp_bias = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        if self.drop_path is None:
            with self.bias_dropout_add_exec_handler():
                output = bias_dropout_add_func(
                    mlp_output,
                    mlp_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)

            # Jit compiled function creates 'view' tensor. This tensor
            # potentially gets saved in the MPU checkpoint function context,
            # which rejects view tensors. While making a viewless tensor here
            # won't result in memory savings (like the data loader, or
            # p2p_communication), it serves to document the origin of this
            # 'view' tensor.
            output = core.utils.make_viewless_tensor(inp = output,
                                              requires_grad = output.requires_grad,
                                              keep_graph = True)

        else:
            out = torch.nn.functional.dropout(mlp_output + mlp_bias,
                                              p=self.hidden_dropout,
                                              training=self.training)
            output = residual + self.drop_path(out)

        return output

    @staticmethod
    def _build_alibi_tensor(max_seq_len, num_attention_heads, batch_size):
        # Based on https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
        """Returns tensor shaped (batch_size * num_attention_heads, 1, max_seq_len)"""

        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = (2 ** (-2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio ** i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                                   :n - closest_power_of_2]

        slopes = torch.Tensor(get_slopes(num_attention_heads))
        alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0).expand(
            num_attention_heads, -1, -1)

        #Select the part of the tensor that corresponds to our tensor parallel index.
        tp_world_size = mpu.get_tensor_model_parallel_world_size()
        tp_index = mpu.get_tensor_model_parallel_rank()
        alibi = alibi.reshape((tp_world_size, -1, *alibi.shape[1:]))[tp_index]

        alibi = alibi.repeat(batch_size, 1, 1)
        return alibi

class NoopTransformerLayer(MegatronModule):
    """A single 'no-op' transformer layer.

    The sole purpose of this layer is for when a standalone embedding layer
    is used (i.e., args.standalone_embedding_stage == True). In this case,
    zero transformer layers are assigned when pipeline rank == 0. Additionally,
    when virtual pipeline rank >= 1, zero total model parameters are created
    (virtual rank 0 contains the input embedding). This results in the model's
    input and output tensors being the same, which causes an error when
    performing certain memory optimiations on the output tensor (e.g.,
    deallocating it). Thus, this layer disconnects the input from the output
    via a clone. Since ranks containing a no-op layer are generally under-
    utilized (both compute and memory), there's no worry of any performance
    degredation.
    """

    def __init__(self, layer_number):
        super().__init__()
        self.layer_number = layer_number

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                inference_params=None):
        return hidden_states.clone()


def _get_num_layers(args, is_encoder_and_decoder_model):
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
            num_ranks_in_decoder = \
                args.transformer_pipeline_model_parallel_size \
                - num_ranks_in_encoder
            assert args.num_layers % num_ranks_in_encoder == 0, \
                'num_layers (%d) must be divisible by number' \
                ' of ranks given to encoder (%d)' \
                % (args.num_layers, num_ranks_in_encoder)
            assert args.num_layers % num_ranks_in_decoder == 0, \
                'num_layers (%d) must be divisible by number ' \
                'of ranks given to decoder (%d)' \
                % (args.num_layers, num_ranks_in_decoder)
            if mpu.is_pipeline_stage_before_split():
                num_layers = (0 if args.standalone_embedding_stage
                              and mpu.get_pipeline_model_parallel_rank() == 0
                              else args.num_layers // num_ranks_in_encoder)
            else:
                num_layers = args.num_layers // num_ranks_in_decoder
        else:
            assert args.num_layers %\
                   args.transformer_pipeline_model_parallel_size ==\
                   0, 'num_layers must be divisible by' \
                      ' transformer_pipeline_model_parallel_size'

            # When a standalone embedding stage is used, all transformer layers
            # are divided among pipeline rank >= 1, while on pipeline rank 0,
            # ranks either contain the input
            # embedding layer (virtual pp rank 0),
            # or no layers at all (virtual pp rank >= 1).
            num_layers = (0 if args.standalone_embedding_stage
                          and mpu.get_pipeline_model_parallel_rank() == 0 else
                          args.num_layers //
                          args.transformer_pipeline_model_parallel_size)
    else:
        num_layers = args.num_layers
    return num_layers


class ParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(self, init_method, output_layer_init_method,
                 layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                  post_layer_norm=True,
                 pre_process=True, post_process=True,
                 drop_path_rate=0.0):
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
        self.drop_path_rate = drop_path_rate

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

        self.drop_path_rates = [rate.item() for rate in torch.linspace(0, self.drop_path_rate, args.num_layers)]

        # Transformer layers.
        def build_layer(layer_number):
            return ParallelTransformerLayer(
                init_method,
                output_layer_init_method,
                layer_number,
                layer_type=layer_type,
                self_attn_mask_type=self_attn_mask_type,
                drop_path_rate=self.drop_path_rates[layer_number - 1])
        if args.virtual_pipeline_model_parallel_size is not None:
            assert args.num_layers % args.virtual_pipeline_model_parallel_size == 0, \
                'num_layers_per_stage must be divisible by ' \
                'virtual_pipeline_model_parallel_size'
            assert args.model_type != ModelType.encoder_and_decoder
            # Number of layers in each model chunk is the number of layers in the stage,
            # divided by the number of model chunks in a stage.
            self.num_layers = self.num_layers // args.virtual_pipeline_model_parallel_size
            # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0]  [2]  [4]  [6]
            # Stage 1: [1]  [3]  [5]  [7]
            # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]  [4, 5]
            # Stage 1: [2, 3]  [6, 7]
            offset = mpu.get_virtual_pipeline_model_parallel_rank() * (
                args.num_layers // args.virtual_pipeline_model_parallel_size) + \
                (mpu.get_pipeline_model_parallel_rank() * self.num_layers)
        else:
            # Each stage gets a contiguous set of layers.
            if args.model_type == ModelType.encoder_and_decoder and \
                    mpu.get_pipeline_model_parallel_world_size() > 1:
                pipeline_rank = mpu.get_pipeline_model_parallel_rank()
                if layer_type == LayerType.encoder:
                    offset = pipeline_rank * self.num_layers
                else:
                    num_ranks_in_enc = args.pipeline_model_parallel_split_rank
                    offset = (pipeline_rank - num_ranks_in_enc) * self.num_layers
            else:
                offset = mpu.get_pipeline_model_parallel_rank() * self.num_layers

        if self.num_layers == 0:
            # When a standalone embedding stage is used (e.g.,
            # args.standalone_embedding_stage == True), virtual pipeline ranks
            # on pipeline rank 0 will have zero transformer layers assigned to
            # them. This results in the model's input and output tensors to be
            # the same, which will cause failure for certain output tensor
            # optimizations (e.g., pipeline output deallocation). To remedy
            # this, we assign a 'no-op' layer on these ranks, which will
            # disconnect the input tensor from the output tensor.
            self.num_layers = 1
            self.layers = torch.nn.ModuleList([ NoopTransformerLayer(1) ])
        else:
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

    def _checkpointed_forward(self, hidden_states, attention_mask,
                              encoder_output, enc_dec_attn_mask):
        """Forward method with activation checkpointing."""
        def custom(start, end):
            def custom_forward(*inputs):
                x_ = inputs[0]
                attention_mask = inputs[1]
                encoder_output = inputs[2]
                enc_dec_attn_mask = inputs[3]
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(x_, attention_mask, encoder_output, enc_dec_attn_mask)
                return x_
            return custom_forward

        if self.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            l = 0
            while l < self.num_layers:
                hidden_states = tensor_parallel.checkpoint(
                    custom(l, l + self.recompute_num_layers),
                    self.distribute_saved_activations,
                    hidden_states, attention_mask, encoder_output, enc_dec_attn_mask)
                l += self.recompute_num_layers

        elif self.recompute_method == 'block':
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            for l in range(self.num_layers):
                if l < self.recompute_num_layers:
                    hidden_states = tensor_parallel.checkpoint(
                        custom(l, l + 1),
                        self.distribute_saved_activations,
                        hidden_states, attention_mask, encoder_output, enc_dec_attn_mask)
                else:
                    hidden_states = custom(l, l + 1)(
                        hidden_states, attention_mask, encoder_output, enc_dec_attn_mask)
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(self, hidden_states, attention_mask,
                 encoder_output=None, enc_dec_attn_mask=None,
                 inference_params=None):
        # hidden_states: [s, b, h]
        timers = get_timers()
        args = get_args()

        if args.transformer_timers: timers("Transformer forward").start()

        # Checks.
        if inference_params:
            assert self.recompute_granularity is None, \
                'inference does not work with activation checkpointing'

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
                hidden_states = self._checkpointed_forward(hidden_states,
                                                           attention_mask,
                                                           encoder_output,
                                                           enc_dec_attn_mask)
            else:
                for index in range(self.num_layers):
                    layer = self._get_layer(index)
                    hidden_states = layer(
                        hidden_states,
                        attention_mask,
                        encoder_output=encoder_output,
                        enc_dec_attn_mask=enc_dec_attn_mask,
                        inference_params=inference_params)

        # Final layer norm.
        if self.post_process and self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        if args.transformer_timers: timers("Transformer forward").stop()

        return hidden_states

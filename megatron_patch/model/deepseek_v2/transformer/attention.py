# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union
import math
import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.utils import divide
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron_patch.model.deepseek_v2.yarn_rotary_pos_embedding import DeepseekV2YarnRotaryEmbedding, \
    apply_rotary_pos_emb, yarn_get_mscale

@dataclass
class SelfAttentionSubmodules:
    linear_q_proj: Union[ModuleSpec, type] = None
    linear_q_a_proj: Union[ModuleSpec, type] = None
    linear_q_b_proj: Union[ModuleSpec, type] = None
    linear_kv_a_proj_with_mqa: Union[ModuleSpec, type] = None
    linear_kv_b_proj: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_a_layernorm: Union[ModuleSpec, type] = None
    kv_a_layernorm: Union[ModuleSpec, type] = None


class Attention(MegatronModule, ABC):
    """Attention layer abstract class.

    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Union[SelfAttentionSubmodules],
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
    ):
        super().__init__(config=config)

        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

        self.num_heads = self.config.num_attention_heads

        # For normal attention without groups, num_query_groups == num_attention_heads,
        # so these two will be the same
        self.query_projection_size = self.config.v_head_dim * self.config.num_attention_heads
        #self.query_projection_size = 196 * self.config.num_attention_heads
        self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = divide(
            self.query_projection_size, self.config.num_attention_heads
        )
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        self.q_head_dim = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim
        self.softmax_scale = self.q_head_dim ** (-0.5)
        mscale = yarn_get_mscale(40, 0.707)
        self.softmax_scale = self.softmax_scale * mscale * mscale

        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
        )

        self.checkpoint_core_attention = self.config.recompute_granularity == 'selective'

        # Output.
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='proj',
        )

        kwargs = {
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
        }

        self.rotary_pos_emb = DeepseekV2YarnRotaryEmbedding(
            self.config.qk_rope_head_dim,
            max_position_embeddings=self.config.max_position_embeddings,
            scaling_factor=self.config.rotary_scaling_factor,
            base=self.config.rotary_base,
            **kwargs,
        )

    def _checkpointed_attention_forward(
        self,
        query,
        key,
        value,
        attention_mask,
        rotary_pos_emb=None,
        attn_mask_type=None,
        packed_seq_params=None,
    ):
        """Forward method with selective activation checkpointing."""

        def custom_forward(*inputs):
            query = inputs[0]
            key = inputs[1]
            value = inputs[2]
            attention_mask = inputs[3]
            attn_mask_type = inputs[5]
            attn_mask_type = AttnMaskType(attn_mask_type.item())
            output_ = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
            return output_

        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type
        attn_mask_type = torch.tensor([attn_mask_type.value], dtype=torch.int)
        hidden_states = tensor_parallel.checkpoint(
            custom_forward,
            False,
            query,
            key,
            value,
            attention_mask,
            rotary_pos_emb,
            attn_mask_type,
        )

        return hidden_states

    @abstractmethod
    def get_query_key_value_tensors(self, hidden_states, key_value_states, position_ids):
        """
        This method needs to be implemented based on whether the derived class
        is "self-attn" or "cross-attn".
        """

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
        position_ids=None
    ):
        # hidden_states: [sq, b, h]
        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        # query: [96, 1, 16, 128], key:[96, 1, 16, 128], value:[96, 1, 16, 128]

        q_len, bsz, _ = hidden_states.size()

        query_states, key_states, value_states = self.get_query_key_value_tensors(hidden_states, key_value_states, position_ids)

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        )

        kv_seq_len = q_len
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attention_mask = attention_mask.to(torch.bfloat16)
            attention_mask[attention_mask > 0] = -3.3895e+38
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        attn_weights = torch.nn.functional.dropout(
            attn_weights, p=self.config.attention_dropout, training=self.training
        )

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(0, 2).transpose(1, 2).contiguous()

        core_attn_out = attn_output.reshape(q_len, bsz, self.num_heads * self.config.v_head_dim)

        # ==================================
        # core attention computation
        # ==================================
        #torch.Size([96, 1, 16384])
        """
        attn_mask_type = self.attn_mask_type
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )

        if packed_seq_params is not None:
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
        """
        # =================
        # Output. [sq, b, h]
        # =================
        # torch.Size([96, 1, 16384])

        output, bias = self.linear_proj(core_attn_out)

        return output, bias


class SelfAttention(Attention):
    """Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
        )

        if self.config.q_lora_rank is None:

            self.linear_q_proj = build_module(
                submodules.linear_q_proj,
                self.config.hidden_size,
                self.num_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
            )

        else:

            self.linear_q_a_proj = build_module(
                submodules.linear_q_a_proj,
                self.config.hidden_size,
                self.config.q_lora_rank,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
            )

            self.linear_q_b_proj = build_module(
                submodules.linear_q_b_proj,
                self.config.q_lora_rank,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
            )

        self.linear_kv_a_proj_with_mqa = build_module(
            submodules.linear_kv_a_proj_with_mqa,
            self.config.hidden_size,
            self.config.kv_lora_rank + self.config.qk_rope_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        )

        self.linear_kv_b_proj = build_module(
            submodules.linear_kv_b_proj,
            self.config.kv_lora_rank,
            self.config.num_attention_heads * (self.q_head_dim - self.config.qk_rope_head_dim + self.config.v_head_dim),
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        )

        if self.config.q_lora_rank is not None:

            self.q_a_layernorm = build_module(
                submodules.q_a_layernorm,
                hidden_size=self.config.q_lora_rank,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )

        self.kv_a_layernorm = build_module(
            submodules.kv_a_layernorm,
            hidden_size=self.config.kv_lora_rank,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None, position_ids=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]

        q_len, bsz, _ = hidden_states.size()
        if self.config.q_lora_rank is not None:
            # [96, 1, 1536]
            q, _ = self.linear_q_a_proj(hidden_states)

            q = self.q_a_layernorm(q)

            # [96, 1, 24576]
            q, _ = self.linear_q_b_proj(q)
        else:
            q, _ = self.linear_q_proj(hidden_states)

        # [96, 1, 128, 192]
        q = q.view(q_len, bsz, self.num_heads, self.q_head_dim)

        # q_nope: [96, 1, 128, 128], q_pe: [96, 1, 128, 64]
        q_nope, q_pe = torch.split(
            q, [self.config.qk_nope_head_dim, self.config.qk_rope_head_dim], dim=-1
        )

        #[96, 1, 576])
        compressed_kv, _ = self.linear_kv_a_proj_with_mqa(hidden_states)

        #compressed_kv:[96, 1, 512], k_pe: [96, 1, 64]
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.config.kv_lora_rank, self.config.qk_rope_head_dim], dim=-1
        )

        #[96, 1, 32768]
        kv, _ = self.linear_kv_b_proj(self.kv_a_layernorm(compressed_kv))

        #[96, 1, 128, 256])
        kv = kv.view(q_len, bsz, self.num_heads, self.config.qk_nope_head_dim + self.config.v_head_dim)

        #k_nope: [96, 1, 128, 128], value_states: [96, 1, 128, 128]
        k_nope, value_states = torch.split(
            kv, [self.config.qk_nope_head_dim, self.config.v_head_dim], dim=-1
        )

        # [96, 1, 128, 128] -> [1, 128, 96, 128]
        value_states = value_states.transpose(0, 1).transpose(1, 2)
        kv_seq_len = value_states.shape[-2]
        #cos: [96, 64], sin:[96, 64]
        cos, sin = self.rotary_pos_emb(value_states, seq_len=kv_seq_len)

        q_pe = q_pe.transpose(0, 1).transpose(1, 2)
        k_pe = k_pe.transpose(0, 1)
        k_pe = k_pe.view(bsz, q_len, 1, self.config.qk_rope_head_dim).transpose(1, 2)

        #q_pe: [1, 128, 96, 64], k_pe:[1, 1, 96, 64]
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        #[1, 128, 96, 192]
        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        #[96, 1, 128, 128] -> [1, 128, 96, 128]
        q_nope = q_nope.transpose(0, 1).transpose(1, 2)
        query_states[:, :, :, : self.config.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.config.qk_nope_head_dim :] = q_pe


        #[1, 128, 96, 192]
        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        # [96, 1, 128, 128] -> [1, 128, 96, 128]
        k_nope = k_nope.transpose(0, 1).transpose(1, 2)
        key_states[:, :, :, : self.config.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.config.qk_nope_head_dim :] = k_pe

        #[1, 128, 96, 192] -> [96, 1, 128, 192]
        #query = query_states.transpose(0, 2).transpose(1, 2)
        #key = key_states.transpose(0, 2).transpose(1, 2)

        #value = torch.zeros_like(value_states.new_empty(bsz, self.num_heads, q_len, self.q_head_dim))
        #value[:, :, :, : self.config.qk_nope_head_dim] = value_states
        #value = value.transpose(0, 2).transpose(1, 2)
        #value = value_states.transpose(0, 2).transpose(1, 2)
        return query_states, key_states, value_states



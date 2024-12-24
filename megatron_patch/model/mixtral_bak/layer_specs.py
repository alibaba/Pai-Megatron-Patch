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

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

from megatron.core.transformer.custom_layers.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp

from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

from .transformer.mlp import MLP, MLPSubmodules
from .transformer.attention import SelfAttention, SelfAttentionSubmodules
from .moe.moe_layer import MoELayer


# Use this spec to use lower level Transformer Engine modules (required for fp8 training)
def get_gpt_layer_with_transformer_engine_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False
) -> ModuleSpec:
    """
    Generates a spec for a GPT transformer layer using Transformer Engine modules.

    Args:
        num_experts: Optional; the number of experts to use in a Mixture of Experts (MoE) setup.
                     If `None`, a dense multi-layer perceptron (MLP) is used instead of MoE.
        moe_grouped_gemm: Optional; if `True`, enables grouped GEMM for MoE operations,
                          which can be more efficient for certain configurations.

    Returns:
        A ModuleSpec object that specifies how to construct a GPT transformer layer with
        the appropriate submodules for self-attention and MLP/MoE using Transformer Engine optimizations.
    """
    mlp = _get_mlp_module_spec(
        use_te=True, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=TENorm if num_experts else IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


# Use this spec for an implementation using only modules in megatron core
def get_gpt_layer_local_spec(num_experts: int = None, moe_grouped_gemm: bool = False) -> ModuleSpec:
    """
    Generates a specification for a GPT transformer layer using only the core modules from Megatron.

    Args:
        num_experts: Optional; the number of experts to use in a Mixture of Experts (MoE) setup.
                     If `None`, a dense multi-layer perceptron (MLP) is used instead of MoE.
        moe_grouped_gemm: Optional; if `True`, enables grouped GEMM for MoE operations,
                          which can be more efficient for certain configurations.

    Returns:
        A ModuleSpec object that specifies how to construct a GPT transformer layer with
        standard Megatron core modules without the lower-level Transformer Engine optimizations.
    """
    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=FusedLayerNorm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=FusedLayerNorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
            },
        ),
    )


# Helper function to get module spec for MLP/MoE
def _get_mlp_module_spec(
    use_te: bool = True, num_experts: int = None, moe_grouped_gemm: bool = False
) -> ModuleSpec:
    """
    Helper function to create a module specification for an MLP or MoE layer.

    Args:
        use_te: Optional; if `True`, uses Transformer Engine (TE) modules for the MLP configuration.
        num_experts: Optional; the number of experts in the MoE configuration. If `None`, a standard MLP is used.
        moe_grouped_gemm: Optional; if `True`, uses grouped GEMM optimization for the MoE configuration.

    Returns:
        A ModuleSpec object that specifies the MLP or MoE layer configuration based on the presence of experts
        and the use of Transformer Engine optimizations.
    """
    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=TELayerNormColumnParallelLinear if use_te else ColumnParallelLinear,
                linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
            ),
        )
    else:
        # SwitchMLP based MoE with modules in megatron core.
        return ModuleSpec(
            module=MoELayer,
            submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear,)
            if not moe_grouped_gemm
            else None,
        )

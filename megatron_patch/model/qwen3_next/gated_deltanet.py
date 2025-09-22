# Copyright (c) 2024 Alibaba PAI and Nvidia Megatron-LM Team.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import warnings
from dataclasses import dataclass, replace
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import ReplicaId, ShardedTensorFactory
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.utils import (
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.utils import deprecate_inference_params, log_single_rank

from megatron.core.ssm.mamba_context_parallel import MambaContextParallel

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update = None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
    from mamba_ssm.ops.triton.ssd_combined import (
        mamba_chunk_scan_combined,
        mamba_split_conv1d_scan_combined,
    )

    HAVE_MAMBA_SSM = True
except ImportError:
    from unittest.mock import MagicMock

    RMSNormGated = MagicMock()
    HAVE_MAMBA_SSM = False

try:
    from einops import rearrange, repeat

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False


logger = logging.getLogger(__name__)


class ExtendedRMSNorm(RMSNormGated):
    """
    RMSNormGated with sharded state dict.
    """

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias not sharded"""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {"weight": 0}, sharded_offsets
        )

class Qwen3NextRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))

        return hidden_states.to(input_dtype)


@dataclass
class MambaMixerSubmodules:
    """
    Contains the module specs for the input and output linear layers.
    """

    in_proj: Union[ModuleSpec, type] = None
    out_proj: Union[ModuleSpec, type] = None


class GatedDeltaNetMixer(MegatronModule):
    """
    Args:
        config: The config of the model.
        submodules: Contains the module specs for the input and output linear layers.
        d_model: The hidden size of the model.
        d_state: The state size of the SSM.
        d_conv: The number of channels in the causal convolution.
        conv_init: The initialization range for the causal convolution weights.
        expand: The expansion factor for the SSM.
        headdim: The hidden size of each attention head.
        ngroups: The number of attention heads.
        A_init_range: The initialization range for the attention weights.
        D_has_hdim: Whether the D parameter has the same number of dimensions as the hidden
            state.
        rmsnorm: Whether to use root mean square normalization.
        norm_before_gate: Whether to apply normalization before the gating mechanism.
        dt_min: The minimum value of the dt parameter.
        dt_max: The maximum value of the dt parameter.
        dt_init: The initialization value of the dt parameter.
        dt_scale: The scaling factor for the dt parameter.
        dt_init_floor: The minimum value of the dt parameter after initialization.
        bias: Whether to use bias in the linear layers.
        conv_bias: Whether to use bias in the causal convolution.
        chunk_size: The chunk size for the fused kernel.
        use_mem_eff_path: Whether to use the memory-efficient path for the Mamba model.
        layer_number: The layer number of this Mamba layer.
        pg_collection: The required process groups to use for tensor model parallel and context
            parallel.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MambaMixerSubmodules,
        d_model,
        d_conv=4,
        conv_init=None,
        expand=2,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=128,
        layer_number=None,
        use_mem_eff_path=None,
        d_state=None,
        headdim=None,
        ngroups=None,
        pg_collection: ProcessGroupCollection = None,
    ):
        if not HAVE_MAMBA_SSM:
            raise ImportError(
                "MambaSSM is not installed. Please install it with `pip install mamba-ssm`."
            )

        if not HAVE_FLA:
            raise ImportError(
                "FLA is not installed"
            )

        super().__init__(config)
        self.config = config
        self.d_model = d_model
        self.rmsnorm = rmsnorm
        assert pg_collection is not None, "pg_collection must be provided for MambaMixer"
        self.pg_collection = pg_collection

        self.head_k_dim = self.config.head_k_dim
        self.head_v_dim = self.config.head_v_dim
        self.num_k_heads = self.config.num_k_heads
        self.num_v_heads = self.config.num_v_heads
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        
        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        self.in_proj_qkvz = build_module(
            submodules.in_proj,
            self.d_model,
            projection_size_qkvz,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
            tp_group=self.pg_collection.tp,
        )

        projection_size_ba = self.num_v_heads * 2
        self.in_proj_ba = build_module(
            submodules.in_proj,
            self.d_model,
            projection_size_ba,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
            tp_group=self.pg_collection.tp,
        )

        self.conv_kernel_size = 4
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        self.activation = "silu"
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = Qwen3NextRMSNormGated(self.head_v_dim, eps=1e-6)

        # Assume sequence parallelism: input is partitioned along d_inner and
        # output is partitioned along the sequence dimension
        self.out_proj = build_module(
            submodules.out_proj,
            self.value_dim,
            self.d_model,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="fc2",
            tp_group=self.pg_collection.tp,
        )

    def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvz` and `mixed_ba`.
        """

        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads,
            2 * self.head_k_dim + 2 * self.head_v_dim * self.num_v_heads // self.num_k_heads,
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (self.num_k_heads, 2 * self.num_v_heads // self.num_k_heads)

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)
        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [self.num_v_heads // self.num_k_heads, self.num_v_heads // self.num_k_heads]
        query, key, value, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=3)
        b, a = torch.split(mixed_ba, split_arg_list_ba, dim=3)
        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), value.size(1), -1, self.head_v_dim)
        z = z.reshape(z.size(0), z.size(1), -1, self.head_v_dim)
        b = b.reshape(b.size(0), b.size(1), self.num_v_heads)
        a = a.reshape(a.size(0), a.size(1), self.num_v_heads)
        return query, key, value, z, b, a

    def forward(
        self,
        hidden_states,
        inference_context=None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ):
        """
        hidden_states: (nL, B, D) / (L B D)
        Returns: same shape as hidden_states
        """
        

        seq_len, batch_size, dim = hidden_states.shape

        conv_state, ssm_state = None, None

        #projected_states_qkvz (seqlen, batch_size, dim)
        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
        projected_states_ba, _ = self.in_proj_ba(hidden_states)
        projected_states_qkvz = projected_states_qkvz.permute(1, 0, 2).contiguous()
        projected_states_ba = projected_states_ba.permute(1, 0, 2).contiguous()

        query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba)
        query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

        mixed_qkv = torch.cat((query, key, value), dim=-1)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        mixed_qkv = causal_conv1d_fn(
            x=mixed_qkv,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
            seq_idx=None,
        )
        
        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )
        query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)


        core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)
        core_attn_out = core_attn_out.permute(1, 0, 2).contiguous()
        
        out, out_bias = self.out_proj(core_attn_out)

        """
        zxBCdt = self.cp.pre_conv_ssm(zxBCdt)

        # transpose: l b pd --> b l pd
        zxBCdt = rearrange(zxBCdt, "l b d -> b l d").contiguous()

        # (nheads_local_tpcp)
        A = -torch.exp(self.cp.get_A_log().float())

        if self.use_mem_eff_path and inference_context is None:
            assert ssm_state is None

            # TODO(duncan): Can this code be removed?
            if self.conv1d.bias is not None:
                self.conv1d.bias.data_ptr()

            y = mamba_split_conv1d_scan_combined(
                zxBCdt,
                rearrange(self.cp.get_conv1d_weight(), "d 1 w -> d w"),
                self.cp.get_conv1d_bias(),
                self.cp.get_dt_bias().float(),
                A,
                D=(
                    rearrange(self.cp.get_D().float(), "(h p) -> h p", p=self.headdim)
                    if self.D_has_hdim
                    else self.cp.get_D()
                ),
                chunk_size=self.chunk_size,
                activation=self.activation,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.cp.ngroups_local_tpcp,
                norm_before_gate=self.norm_before_gate,
            )

            y = rearrange(y, "b l d -> l b d").contiguous()
            y = self.cp.post_conv_ssm(y)

            if self.rmsnorm:
                y = self.norm(y)
        else:
            # This path is always used for the inference prefill phase.
            # `mamba_split_conv1d_scan_combined`, used in the other branch above, reduces the size
            # of forward activations stored for backprop, which reduces memory pressure during
            # training, and does not provide increased speed in the forward direction.
            z, xBC, dt = torch.split(
                zxBCdt,
                [
                    self.cp.d_inner_local_tpcp,
                    self.cp.d_inner_local_tpcp + 2 * self.cp.ngroups_local_tpcp * self.d_state,
                    self.cp.nheads_local_tpcp,
                ],
                dim=-1,
            )

            # transpose: b l pd --> b pd l
            xBC = rearrange(xBC, "b l d -> b d l").contiguous()

            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(
                    F.pad(xBC, (self.d_conv - xBC.shape[-1], 0))
                )  # Update state (B D W)

            seqlen = xBC.size(2)
            if causal_conv1d_fn is None:
                xBC = self.act(self.cp.conv1d(xBC)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                xBC = causal_conv1d_fn(
                    x=xBC,
                    weight=rearrange(self.cp.get_conv1d_weight(), "d 1 w -> d w"),
                    bias=self.cp.get_conv1d_bias(),
                    activation=self.activation,
                )

            # transpose b pd l --> b l pd
            xBC = rearrange(xBC, "b d l ->  b l d").contiguous()

            x, B, C = torch.split(
                xBC,
                [
                    self.cp.d_inner_local_tpcp,
                    self.cp.ngroups_local_tpcp * self.d_state,
                    self.cp.ngroups_local_tpcp * self.d_state,
                ],
                dim=-1,
            )

            # TODO Vijay: fuse most of the transposes with the GEMMS
            x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim).contiguous()
            dt = dt.contiguous()
            B = rearrange(B, "b l (g n) -> b l g n", n=self.d_state).contiguous()
            C = rearrange(C, "b l (g n) -> b l g n", n=self.d_state).contiguous()
            z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim).contiguous()

            # If `rmsnorm == False`, then the norm inside `mamba_chunk_scan_combined` will be used.
            # In this case, if `cp_size > 1` then that norm could be performed on less heads than if
            # `cp_size == 1` (groups of heads can be sharded across CP ranks), which would be
            # mathematically incorrect, and potentially arithmetically unstable.
            assert (
                self.cp.cp_size == 1 or self.rmsnorm
            ), "Context parallel not supported for use_mem_eff_path==False and rmsnorm==False"

            y = mamba_chunk_scan_combined(
                x,
                dt,
                A,
                B,
                C,
                self.chunk_size,
                D=(
                    rearrange(self.cp.get_D().float(), "(h p) -> h p", p=self.headdim)
                    if self.D_has_hdim
                    else self.cp.get_D()
                ),
                z=z if not self.rmsnorm else None,
                dt_bias=self.cp.get_dt_bias().float(),
                dt_softplus=True,
                return_final_states=ssm_state is not None,
            )

            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)

            y = rearrange(y, "b l h p -> l b (h p)").contiguous()
            y = self.cp.post_conv_ssm(y)

            if self.rmsnorm:
                z = rearrange(z, "b l h p -> l b (h p)").contiguous()
                z = self.cp.post_conv_ssm(z)
                y = self.norm(y, z)

        out, out_bias = self.out_proj(y)
        """
        
        return out, out_bias

    def step(self, hidden_states, conv_state, ssm_state):
        """
        Performs inference step for decoding
        """
        # assert self.ngroups_local_tp == 1, "Only support ngroups=1 for inference for now"
        dtype = hidden_states.dtype
        assert hidden_states.shape[0] == 1, "Only support decoding with 1 token at a time for now"

        # l b d --> b d
        hidden_states = hidden_states.squeeze(0)

        #  b d_model --> b p(2d)
        zxBCdt, _ = self.in_proj(hidden_states)

        assert self.cp.cp_size == 1, "Context parallel not supported for Mamba inferenece decode"

        z, xBC, dt = torch.split(
            zxBCdt,
            [
                self.d_inner_local_tp,
                self.d_inner_local_tp + 2 * self.ngroups_local_tp * self.d_state,
                self.nheads_local_tp,
            ],
            dim=-1,
        )

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(
                conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
            )  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x, B, C = torch.split(
            xBC,
            [
                self.d_inner_local_tp,
                self.ngroups_local_tp * self.d_state,
                self.ngroups_local_tp * self.d_state,
            ],
            dim=-1,
        )
        A = -torch.exp(self.A_log.float())

        # SSM step
        if selective_state_update is None:
            if self.ngroups_local_tp > 1:
                B = rearrange(B, "b (g n) -> b g n", n=self.d_state)
                C = rearrange(C, "b (g n) -> b g n", n=self.d_state)
                B = repeat(
                    B, "b g n -> b (g h) n", h=self.d_inner_local_tp // self.ngroups_local_tp
                )
                C = repeat(
                    C, "b g n -> b (g h) n", h=self.d_inner_local_tp // self.ngroups_local_tp
                )

                dt = repeat(dt, "b h -> b (h p)", p=self.headdim)
                dt_bias = repeat(self.dt_bias, "h -> (h p)", p=self.headdim)
                A = repeat(A, "h -> (h p) n", p=self.headdim, n=self.d_state)
                D = repeat(self.D, "h -> (h p)", p=self.headdim)

                dt = F.softplus(dt + dt_bias.to(dtype=dt.dtype))
                dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))

                dB_x = torch.einsum("bd,bdn,bd->bdn", dt, B, x)
                ssm_state.copy_(
                    ssm_state * rearrange(dA, "b (h p) n -> b h p n", p=self.headdim)
                    + rearrange(dB_x, "b (h p) n -> b h p n", p=self.headdim)
                )

                y = torch.einsum(
                    "bdn,bdn->bd",
                    rearrange(ssm_state.to(dtype), "b h p n -> b (h p) n", p=self.headdim),
                    C,
                )
                y = y + D.to(dtype) * x
                if not self.rmsnorm:
                    y = y * self.act(z)  # (B D)
            else:
                # Discretize A and B (b (g n))
                dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
                dA = torch.exp(dt * A)
                x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
                dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
                ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
                y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
                y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
                y = rearrange(y, "b h p -> b (h p)")
                if not self.rmsnorm:
                    y = y * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups_local_tp)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups_local_tp)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state,
                x_reshaped,
                dt,
                A,
                B,
                C,
                D,
                z=z if not self.rmsnorm else None,
                dt_bias=dt_bias,
                dt_softplus=True,
            )
            y = rearrange(y, "b h p -> b (h p)")

        if self.rmsnorm:
            y = self.norm(y, z)

        # b pd --> b d
        out, out_bias = self.out_proj(y)
        return out.unsqueeze(0), out_bias, conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        """
        allocate inference cache
        """
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.conv1d.weight.shape[0], self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size,
            self.nheads_local_tp,
            self.headdim,
            self.d_state,
            device=device,
            dtype=ssm_dtype,
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_context, batch_size, *, inference_params=None):
        """Initializes or retrieves the SSM state tensors from the cache.

        At the start of any inference (at the prefill step), if there is no cache or if the
        cached batch size has changed, then new tensors are initialized and stored in the cache.
        Otherwise the existing tensors are retrieved from the cache and zeroed out.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        assert inference_context is not None
        assert self.layer_number is not None
        if (
            self.layer_number not in inference_context.key_value_memory_dict
            or batch_size != self.cached_batch_size
        ):
            conv_state = torch.zeros(
                batch_size,
                self.conv1d.weight.shape[0],
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.nheads_local_tp,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_context.key_value_memory_dict[self.layer_number] = (conv_state, ssm_state)
            self.cached_batch_size = batch_size
        else:
            conv_state, ssm_state = inference_context.key_value_memory_dict[self.layer_number]
            # TODO: Remove reference to `inference_context.sequence_len_offset` for dynamic batching
            if inference_context.sequence_len_offset == 0:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Provide a sharded state dictionary for distributed checkpointing."""
        sharded_state_dict = {}
        # Parameters
        self._save_to_state_dict(sharded_state_dict, "", keep_vars=True)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict,
            prefix,
            tensor_parallel_layers_axis_map={
                "A_log": 0,
                "dt_bias": 0,
                "D": 0,
            },  # parameters sharded across TP
            sharded_offsets=sharded_offsets,
        )
        # Submodules
        for name, module in self.named_children():
            if name == "conv1d":
                # Add TP sharding for Conv1d
                module_sd = module.state_dict(prefix="", keep_vars=True)
                module_sharded_sd = make_sharded_tensors_for_checkpoint(
                    module_sd, f"{prefix}{name}.", {f"weight": 0, f"bias": 0}, sharded_offsets
                )

            else:
                module_sharded_sd = sharded_state_dict_default(
                    module, f"{prefix}{name}.", sharded_offsets, metadata
                )

            sharded_state_dict.update(module_sharded_sd)

        # At this point the TP sharding is correctly defined for each tensor, but some of the
        # tensors must be additionally split into separate parts
        in_proj_dim = (
            self.d_inner_local_tp * 2
            + 2 * self.ngroups_local_tp * self.d_state
            + self.nheads_local_tp
        )
        assert sharded_state_dict[f"{prefix}in_proj.weight"].data.size(0) == in_proj_dim, (
            in_proj_dim,
            sharded_state_dict[f"{prefix}in_proj.weight"],
        )

        sharded_state_dict[f"{prefix}in_proj.weight"] = _split_tensor_factory(
            sharded_state_dict[f"{prefix}in_proj.weight"],
            [
                self.d_inner_local_tp,
                self.d_inner_local_tp,
                self.ngroups_local_tp * self.d_state,
                self.ngroups_local_tp * self.d_state,
                self.nheads_local_tp,
            ],
            ["z", "x", "B", "C", "dt"],
            0,
        )

        conv_dim = self.d_inner_local_tp + 2 * self.ngroups_local_tp * self.d_state
        assert sharded_state_dict[f"{prefix}conv1d.weight"].data.size(0) == conv_dim, (
            conv_dim,
            sharded_state_dict[f"{prefix}conv1d.weight"],
        )
        assert sharded_state_dict[f"{prefix}conv1d.bias"].data.size(0) == conv_dim, (
            conv_dim,
            sharded_state_dict[f"{prefix}conv1d.bias"],
        )

        for conv_layer_name in ["conv1d.weight", "conv1d.bias"]:
            sharded_state_dict[f"{prefix}{conv_layer_name}"] = _split_tensor_factory(
                sharded_state_dict[f"{prefix}{conv_layer_name}"],
                [
                    self.d_inner_local_tp,
                    self.ngroups_local_tp * self.d_state,
                    self.ngroups_local_tp * self.d_state,
                ],
                ["x", "B", "C"],
                0,
            )

        return sharded_state_dict


def _split_tensor_factory(
    orig_sh_ten: ShardedTensor, split_sections: List[int], split_names: List[str], split_dim: int
) -> ShardedTensorFactory:
    """Builds a factory that splits a given ShardedTensor into several independent chunks."""
    assert isinstance(orig_sh_ten, ShardedTensor), type(orig_sh_ten)
    orig_sh_ten_no_data = orig_sh_ten.without_data()  # remove `data` reference

    if sum(split_sections) != orig_sh_ten_no_data.local_shape[split_dim]:
        raise ValueError(
            f"Split sections must cover the whole dimension size, "
            f"got {split_sections=} vs dimensions size "
            f"{orig_sh_ten_no_data.local_shape[split_dim]}"
        )

    assert not isinstance(
        split_sections, int
    ), "Splitting into predefined section sizes is supported (`split_sections` must be a list)"
    assert len(split_sections) == len(split_names), (len(split_sections), len(split_names))

    @torch.no_grad()
    def sh_ten_build_fn(
        key: str, t: torch.Tensor, replica_id: ReplicaId, flattened_range: Optional[slice]
    ):
        factory_sh_ten = replace(
            orig_sh_ten_no_data,
            key=key,
            data=t,
            dtype=t.dtype,
            replica_id=replica_id,
            flattened_range=flattened_range,
        )

        chunk_sh_tens = []
        split_start = 0
        for split_size, split_name in zip(split_sections, split_names):
            split_chunks = factory_sh_ten.narrow(split_dim, split_start, split_size)
            for sh_ten in split_chunks:
                sh_ten.key = f"{sh_ten.key}.{split_name}"
            chunk_sh_tens.extend(split_chunks)
            split_start += split_size

        assert split_start == orig_sh_ten_no_data.local_shape[split_dim], (
            split_start,
            orig_sh_ten_no_data.local_shape[split_dim],
        )
        assert sum(sh_ten.data.numel() for sh_ten in chunk_sh_tens) == t.numel(), (
            chunk_sh_tens,
            t.shape,
        )
        return chunk_sh_tens

    @torch.no_grad()
    def sh_ten_merge_fn(sub_state_dict):
        return torch.cat(sub_state_dict)

    return ShardedTensorFactory(
        orig_sh_ten.key, orig_sh_ten.data, sh_ten_build_fn, sh_ten_merge_fn, orig_sh_ten.replica_id
    )


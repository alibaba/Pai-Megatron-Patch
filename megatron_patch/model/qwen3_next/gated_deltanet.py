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
        norm_before_gate=True,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        bias=False,
        conv_bias=False,
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
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        assert pg_collection is not None, "pg_collection must be provided for MambaMixer"
        self.pg_collection = pg_collection

        self.head_k_dim = self.config.head_k_dim
        self.head_v_dim = self.config.head_v_dim
        self.num_k_heads = self.config.num_k_heads
        self.num_v_heads = self.config.num_v_heads
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.d_state = self.head_k_dim
        self.headdim = self.head_v_dim
        self.ngroups = self.num_k_heads
        self.nheads = self.num_v_heads
        self.d_inner = self.nheads * self.headdim

        
        tp_size = self.pg_collection.tp.size()

        self.nheads_local_tp = self.nheads // tp_size
        self.d_inner_local_tp = self.d_inner // tp_size
        self.ngroups_local_tp = self.ngroups // tp_size
 
        # Assume sequence parallelism: input is already partitioned along the sequence dimension
        self.in_proj = build_module(
            submodules.in_proj,
            self.d_model,
            self.d_inner * 2 + 2 * self.ngroups * self.d_state + self.nheads * 2,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
            tp_group=self.pg_collection.tp,
        )

        conv_dim = self.d_inner_local_tp + 2 * self.ngroups_local_tp * self.d_state  # x B C
        with get_cuda_rng_tracker().fork():
            # weight shape: [conv_dim, 1, d_conv]
            # bias shape: [conv_dim]
            self.conv1d = nn.Conv1d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=conv_dim,
                padding=d_conv - 1,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
            setattr(self.conv1d.weight, "tensor_model_parallel", True)
            if conv_bias:
                setattr(self.conv1d.bias, "tensor_model_parallel", True)

            if self.conv_init is not None:
                nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.activation = "silu"
        self.act = nn.SiLU()

        with get_cuda_rng_tracker().fork():
            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            self.dt_bias = nn.Parameter(torch.ones(self.nheads_local_tp))
            # Our initialization would set all Linear.bias to zero,
            # need to mark this one as _no_reinit
            self.dt_bias._no_reinit = True
            # Just to be explicit. Without this we already don't
            # put wd on dt_bias because of the check
            # name.endswith("bias") in param_grouping.py
            self.dt_bias._no_weight_decay = True
            setattr(self.dt_bias, "tensor_model_parallel", True)

            # A parameter
            A = torch.empty(self.nheads_local_tp).uniform_(0, 16)
            self.A_log = nn.Parameter(torch.log(A))
            self.A_log._no_weight_decay = True
            setattr(self.A_log, "tensor_model_parallel", True)

        # D "skip" parameter
        self.D = None

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(
                self.config.head_v_dim,
                eps=self.config.layernorm_epsilon,
                norm_before_gate=self.norm_before_gate, # True
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )

        # Assume sequence parallelism: input is partitioned along d_inner and
        # output is partitioned along the sequence dimension
        self.out_proj = build_module(
            submodules.out_proj,
            self.d_inner,
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

        # Regarding `conv1d`.{`weight`, `bias`}, `dt_bias`, `A_log`, and `D`: these are the
        # trainable variables for the current tensor parallel rank, with each tensor parallel rank
        # having indepdendent trainable variables. All context parallel ranks in a tensor parallel
        # rank store the same trainable variables, but only use and update their unique/independent
        # slice of them.
        self.cp = MambaContextParallel(
            cp_group=self.pg_collection.cp,
            d_inner_local_tp=self.d_inner_local_tp,
            nheads_local_tp=self.nheads_local_tp,
            ngroups_local_tp=self.ngroups_local_tp,
            d_state=self.d_state,
            conv1d_cp1=self.conv1d,
            dt_bias_cp1=self.dt_bias,
            A_log_cp1=self.A_log,
            D_cp1=self.D,
            D_has_hdim=self.D_has_hdim,
        )

    def forward(
        self,
        hidden_states,
        inference_context=None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ):

        
        seq_len, batch_size, dim = hidden_states.shape

        zVQKba, _ = self.in_proj(hidden_states)

        #zVQKba = self.cp.pre_conv_ssm(zVQKba)

        zVQKba = rearrange(zVQKba, "l b d -> b l d").contiguous()

        z, VQK, ba = torch.split(
            zVQKba,
            [
                self.cp.d_inner_local_tpcp,
                self.cp.d_inner_local_tpcp + 2 * self.cp.ngroups_local_tpcp * self.d_state,
                self.cp.nheads_local_tpcp*2,
            ],
            dim=-1,
        )

        VQK = rearrange(VQK, "b l d -> b d l").contiguous()

        VQK = causal_conv1d_fn(
            x=VQK,
            weight=rearrange(self.cp.get_conv1d_weight(), "d 1 w -> d w"),
            bias=self.cp.get_conv1d_bias(),
            activation=self.activation,
        )
        

        VQK= rearrange(VQK, "b d l ->  b l d").contiguous()

        value, query, key = torch.split(
                VQK,
                [
                    self.cp.d_inner_local_tpcp,
                    self.cp.ngroups_local_tpcp * self.d_state,
                    self.cp.ngroups_local_tpcp * self.d_state,
                ],
                dim=-1,
        )

        b, a= torch.split(
                ba,
                [
                    self.cp.nheads_local_tpcp,
                    self.cp.nheads_local_tpcp,

                ],
                dim=-1,
        )

        z = z.reshape(z.shape[0], z.shape[1], -1, self.head_k_dim)
        query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)


        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.cp.nheads_local_tpcp // self.cp.ngroups_local_tpcp > 1:
            query = query.repeat_interleave(self.cp.nheads_local_tpcp // self.cp.ngroups_local_tpcp, dim=2)
            key = key.repeat_interleave(self.cp.nheads_local_tpcp // self.cp.ngroups_local_tpcp, dim=2)

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

        if self.rmsnorm:
            #z = self.cp.post_conv_ssm(z)
            core_attn_out = self.norm(core_attn_out, z)

        y = rearrange(core_attn_out, "b l h p -> l b (h p)").contiguous()
        #y = self.cp.post_conv_ssm(y)

        out, out_bias = self.out_proj(y)

        return out, out_bias

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
            + self.nheads_local_tp * 2
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
                self.nheads_local_tp,
            ],
            ["z", "V", "Q", "K", "b", "a"],
            0,
        )

        conv_dim = self.d_inner_local_tp + 2 * self.ngroups_local_tp * self.d_state
        assert sharded_state_dict[f"{prefix}conv1d.weight"].data.size(0) == conv_dim, (
            conv_dim,
            sharded_state_dict[f"{prefix}conv1d.weight"],
        )


        for conv_layer_name in ["conv1d.weight"]:
            sharded_state_dict[f"{prefix}{conv_layer_name}"] = _split_tensor_factory(
                sharded_state_dict[f"{prefix}{conv_layer_name}"],
                [
                    self.d_inner_local_tp,
                    self.ngroups_local_tp * self.d_state,
                    self.ngroups_local_tp * self.d_state,
                ],
                ["V", "Q", "K"],
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

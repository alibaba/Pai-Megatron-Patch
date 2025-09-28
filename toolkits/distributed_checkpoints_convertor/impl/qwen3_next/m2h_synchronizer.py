# Copyright (c) 2025 Alibaba PAI Team.
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
import logging

import torch
from typing import Dict
from general.m2h_synchronizer import MG2HFSynchronizer as _MG2HFSynchronizer
from general.synchronizer import ParamType

class MG2HFSynchronizer(_MG2HFSynchronizer):
    # TODO: to be refactored to Hybrid Model convertor
    def __init__(self, load_dir, model_provider_func=None):
        super().__init__(load_dir, model_provider_func)
        self.layout = self.get_hybrid_layout()
        assert self.tp_size == 1, "Currently MCore2HF conversion for Qwen3-Next is only available with TP 1."

    def get_hybrid_layout(self) -> str:
        assert self.args.hybrid_override_pattern is not None
        return self.args.hybrid_override_pattern


    def sync_params(self, mg_model = None, hf_model = None):
        # assume TE backend
        if self.args.transformer_impl != "transformer_engine":
            raise NotImplementedError("Currently only TE model is implemented.")
        
        if mg_model is None:
            mg_model = self._mgmodel
        if hf_model is None:
            hf_model = self._hfmodel

        if mg_model.pre_process:
            self.set_preprocess_state(mg_model=mg_model, hf_model=hf_model)
        
        if mg_model.post_process:
            self.set_postprocess_state(mg_model=mg_model, hf_model=hf_model, is_mamba=True)

        for mg_layer_id, global_mg_layer_id in self._build_pipeline_parallel_mapping().items():
            hf_layer_id  = global_mg_layer_id // 2
            if (
                self.tp_rank == 0 and 
                self.ep_rank == 0 and 
                self.etp_rank == 0 and
                global_mg_layer_id % 2 == 0
            ):
                logging.info(f"Converting layer {hf_layer_id}")

            layer = mg_model.decoder.layers[mg_layer_id]
            hf_layer = hf_model.model.layers[hf_layer_id]

            if self.layout[global_mg_layer_id] == 'M':
                # Mamba layer
                self.set_mamba_layer_state(layer.mixer, hf_layer.linear_attn)
                self.copy(layer.mixer.in_proj.layer_norm_weight, hf_layer.input_layernorm.weight)
            elif self.layout[global_mg_layer_id] == '-':
                # transformer_layer of MLP
                self.set_moe_layer_state(layer.mlp, hf_layer.mlp)
                self.copy(layer.pre_mlp_layernorm.weight, hf_layer.post_attention_layernorm.weight)
            elif self.layout[global_mg_layer_id] == '*':
                # transformer_layer of Attention
                self.set_gated_selfattn_state(layer.self_attention, hf_layer.self_attn)
                self.copy(layer.self_attention.linear_qgkv.layer_norm_weight, hf_layer.input_layernorm.weight)
            else:
                raise ValueError(f"Unrecognized layer type {self.layout[global_mg_layer_id]} in {self.layout}")

    def set_mamba_layer_state(self, mixer, hf_mixer):
        Nk, Nv, Dk, Dv = (
            hf_mixer.num_k_heads,
            hf_mixer.num_v_heads,
            hf_mixer.head_k_dim,
            hf_mixer.head_v_dim
        )
        split_size_list = [
            Dv * Nv // self.tp_size, 
            Dv * Nv // self.tp_size, 
            Dk * Nk // self.tp_size, 
            Dk * Nk // self.tp_size, 
            Nv // self.tp_size, 
            Nv // self.tp_size
        ]
        z, v, q, k, b, a = torch.split(
            mixer.in_proj.weight,
            split_size_list,
            dim=0
        )
        in_proj_qkvz_weight = torch.cat([
            q.reshape(Nk // self.tp_size, Dk, -1), 
            k.reshape(Nk // self.tp_size, Dk, -1), 
            v.reshape(Nk // self.tp_size, Dv * Nv // Nk, -1), 
            z.reshape(Nk // self.tp_size, Dv * Nv // Nk, -1), 
        ], dim=1)
        self.copy(in_proj_qkvz_weight, hf_mixer.in_proj_qkvz.weight, param_type=ParamType.QKV_W)

        in_proj_ba_weight = torch.cat([
            b.reshape(Nk // self.tp_size, Nv // Nk, -1), 
            a.reshape(Nk // self.tp_size, Nv // Nk, -1), 
        ], dim=1)
        self.copy(in_proj_ba_weight, hf_mixer.in_proj_ba.weight, param_type=ParamType.QKV_W)

        self.copy(mixer.dt_bias, hf_mixer.dt_bias, param_type=ParamType.COLUMN)
        self.copy(mixer.A_log, hf_mixer.A_log, param_type=ParamType.COLUMN)

        # TODO: support TP > 1 if needed
        split_size_list = [
            Nv * Dv, 
            Nk * Dk, 
            Nk * Dk, 
        ]
        conv_v, conv_q, conv_k = torch.split(
            mixer.conv1d.weight, 
            split_size_or_sections=split_size_list, 
            dim=0
        )
        conv1d_weight = torch.cat([
            conv_q, 
            conv_k, 
            conv_v, 
        ], dim=0)
        self.copy(conv1d_weight, hf_mixer.conv1d.weight, param_type=ParamType.UNIQUE)
        
        self.copy(mixer.norm.weight, hf_mixer.norm.weight, param_type=ParamType.UNIQUE)
        self.copy(mixer.out_proj.weight, hf_mixer.out_proj.weight, param_type=ParamType.ROW)

    def _build_pipeline_parallel_mapping(self) -> Dict[int, int]:
        remained_num_layers = self.args.num_layers
        remained_stages = self.pp_size
        pp_layers_per_stage = [remained_num_layers // remained_stages] * remained_stages

        pp_mapping = {
            i: v for i, v in enumerate(
                range(
                    sum(pp_layers_per_stage[:self.pp_rank]), 
                    sum(pp_layers_per_stage[:self.pp_rank + 1])
                )
            )
        }
        return pp_mapping

    def set_gated_selfattn_state(self, attn, hf_attn):
        '''Set gated self-attention params.'''
        # Reshape loaded weights.
        tp = self.tp_size
        num_heads = self.args.num_attention_heads
        num_query_groups = (self.args.num_query_groups if self.args.group_query_attention else self.args.num_attention_heads)
        num_querys_per_group = num_heads // num_query_groups
        dim = self.args.kv_channels
        assert num_heads % num_querys_per_group == 0
        # copy qk norm if indeed.
        if self.args.qk_layernorm:
            self.copy(attn.q_layernorm.weight, hf_attn.q_norm.weight)
            self.copy(attn.k_layernorm.weight, hf_attn.k_norm.weight)

        # Copy weights (re-order dimensions for Megatron).
        attn_proj_weight = attn.linear_qgkv.weight.reshape(
            (num_query_groups // tp, (2 + num_querys_per_group*2)*dim, -1)
        )
        (
            q_proj_weight, 
            k_proj_weight, 
            v_proj_weight
        ) = torch.split(attn_proj_weight, [2*num_querys_per_group*dim, dim, dim], dim=1)

        q_proj_weight = q_proj_weight.reshape(num_query_groups // tp, 2, num_querys_per_group, dim, -1).transpose(1, 2).flatten(1, 3)
        self.copy(q_proj_weight, hf_attn.q_proj.weight, param_type=ParamType.QKV_W)
        self.copy(k_proj_weight, hf_attn.k_proj.weight, param_type=ParamType.QKV_W)
        self.copy(v_proj_weight, hf_attn.v_proj.weight, param_type=ParamType.QKV_W)

        self.copy(
            attn.linear_proj.weight,
            hf_attn.o_proj.weight,
            param_type=ParamType.ROW
        )

        # Copy bias
        if self.args.add_qkv_bias:
            attn_proj_bias = attn.linear_qkv.bias.reshape(
                (num_query_groups // tp, (2 + num_querys_per_group * 2)*dim, -1)
            )
            q_proj_bias, k_proj_bias, v_proj_bias = torch.split(
                attn_proj_bias, 
                [2*num_querys_per_group*dim, dim, dim], 
                dim=1
            )
            self.copy(q_proj_bias, hf_attn.q_proj.bias, param_type=ParamType.QKV_B)
            self.copy(k_proj_bias, hf_attn.k_proj.bias, param_type=ParamType.QKV_B)
            self.copy(v_proj_bias, hf_attn.v_proj.bias, param_type=ParamType.QKV_B)
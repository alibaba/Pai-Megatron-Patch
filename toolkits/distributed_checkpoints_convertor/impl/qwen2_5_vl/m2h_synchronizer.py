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
from packaging.version import Version as PkgVersion
import transformers

from general.m2h_synchronizer import MG2HFSynchronizer as _MG2HFSynchronizer
from general.synchronizer import ParamType

class MG2HFSynchronizer(_MG2HFSynchronizer):

    def sync_params(self):
        if PkgVersion(transformers.__version__) >= PkgVersion('4.52.0'):
            super().sync_params(self._mgmodel.language_model, self._hfmodel.model.language_model)
        else:
            super().sync_params(self._mgmodel.language_model, self._hfmodel.model)
        if self._mgmodel.pre_process:
            self.set_vision_model_layer_state(
                self._mgmodel.vision_model,
                self._hfmodel.visual
            )

    def set_vision_model_layer_state(self, vision_model, hf_vision_model):
        self.copy(
            vision_model.patch_embed.proj.weight,
            hf_vision_model.patch_embed.proj.weight
        )
        for layer, hf_layer in zip(
            vision_model.decoder.layers,
            hf_vision_model.blocks
        ):
            self.set_vision_layer_state(layer, hf_layer)
        self.copy(vision_model.decoder.final_layernorm.weight, hf_vision_model.merger.ln_q.weight)
        self.set_merger_mlp_state(vision_model.projection.encoder, hf_vision_model.merger.mlp)

    def set_merger_mlp_state(self, mlp, hf_mlp):
        self.copy(mlp.linear_fc1.weight,hf_mlp[0].weight, param_type=ParamType.COLUMN)
        self.copy(mlp.linear_fc1.bias, hf_mlp[0].bias, param_type=ParamType.COLUMN)
        self.copy(mlp.linear_fc2.weight, hf_mlp[2].weight, param_type=ParamType.ROW)
        self.copy(mlp.linear_fc2.bias, hf_mlp[2].bias)

    def set_vision_layer_state(self, layer, hf_layer):
        self.set_visual_attn_state(layer.self_attention, hf_layer.attn)
        self.copy(layer.self_attention.linear_qkv.layer_norm_weight, hf_layer.norm1.weight)
        if layer.config.normalization == 'LayerNorm':
            self.copy(layer.self_attention.linear_qkv.layer_norm_bias, hf_layer.norm1.bias)

        self.set_mlp_state(layer.mlp, hf_layer.mlp)
        self.copy(layer.mlp.linear_fc1.layer_norm_weight, hf_layer.norm2.weight)
        if layer.config.normalization == 'LayerNorm':
            self.copy(layer.mlp.linear_fc1.layer_norm_bias, hf_layer.norm2.bias)

    def set_visual_attn_state(self, attn, hf_attn):
        '''Set self-attention params.'''
        # Reshape loaded weights.
        tp = self.tp_size
        num_heads = attn.config.num_attention_heads
        num_query_groups = attn.config.num_query_groups
        num_querys_per_group = num_heads // num_query_groups
        dim = attn.config.kv_channels
        assert num_heads % num_querys_per_group == 0

        # Copy weights (re-order dimensions for Megatron).
        attn_proj_weight = attn.linear_qkv.weight.reshape(
            (num_query_groups // tp, 2 + num_querys_per_group, dim, -1)
        ).transpose(0, 1)

        self.copy(
            attn_proj_weight.reshape(-1, dim, hf_attn.qkv.weight.shape[-1]), 
            hf_attn.qkv.weight,
            param_type=ParamType.QKV_W,
        )

        self.copy(
            attn.linear_proj.weight,
            hf_attn.proj.weight,
            param_type=ParamType.ROW
        )
        self.copy(
            attn.linear_proj.bias,
            hf_attn.proj.bias,
        )

        # Copy bias
        if attn.config.add_qkv_bias:
            attn_proj_bias = attn.linear_qkv.bias.reshape(
                (num_query_groups // tp, 2 + num_querys_per_group, dim, -1)
            ).transpose(0, 1)
            self.copy(
                attn_proj_bias.flatten(0, 1), 
                hf_attn.qkv.bias,
                param_type=ParamType.QKV_B,
            )
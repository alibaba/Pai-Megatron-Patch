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
from general.h2m_synchronizer import HF2MGSynchronizer as _HF2MGSynchronizer
from general.synchronizer import ParamType

class HF2MGSynchronizer(_HF2MGSynchronizer):

    def sync_params(self):
        super().sync_params(self._mgmodel.language_model, self._hfmodel)
        if self._mgmodel.pre_process:
            self.set_vision_model_layer_state(
                self._mgmodel.vision_model,
                self._hfmodel.visual
            )

    def set_vision_model_layer_state(self, vision_model, hf_vision_model):
        self.copy(
            hf_vision_model.patch_embed.proj.weight,
            vision_model.patch_embed.proj.weight
        )
        for layer, hf_layer in zip(
            vision_model.decoder.layers,
            hf_vision_model.blocks
        ):
            self.set_vision_layer_state(layer, hf_layer)
        self.copy(hf_vision_model.merger.ln_q.weight, vision_model.decoder.final_layernorm.weight)

        self.copy(hf_vision_model.merger.mlp[0].weight, vision_model.projection.encoder.linear_fc1.weight, param_type=ParamType.COLUMN)
        self.copy(hf_vision_model.merger.mlp[0].bias, vision_model.projection.encoder.linear_fc1.bias, param_type=ParamType.COLUMN)
        self.copy(hf_vision_model.merger.mlp[2].weight, vision_model.projection.encoder.linear_fc2.weight, param_type=ParamType.ROW)
        self.copy(hf_vision_model.merger.mlp[2].bias, vision_model.projection.encoder.linear_fc2.bias)

    def set_vision_layer_state(self, layer, hf_layer):
        self.set_visual_attn_state(layer.self_attention, hf_layer.attn)
        self.copy(hf_layer.norm1.weight, layer.self_attention.linear_qkv.layer_norm_weight)
        
        self.set_mlp_state(layer.mlp, hf_layer.mlp)
        self.copy(hf_layer.norm2.weight, layer.mlp.linear_fc1.layer_norm_weight)
    
    def set_visual_attn_state(self, attn, hf_attn):
        '''Set self-attention params.'''
        # Reshape loaded weights.
        num_heads = attn.config.num_attention_heads
        num_query_groups = attn.config.num_query_groups
        num_querys_per_group = num_heads // num_query_groups
        dim = attn.config.kv_channels
        assert num_heads % num_querys_per_group == 0

        # Copy weights (re-order dimensions for Megatron).
        if self.dryrun:
            attn_proj_weight = attn.linear_qkv.weight
        else:
            attn_proj_weight = self.load_tensor(
                hf_attn.qkv.weight
            ).view(num_querys_per_group + 2, num_query_groups, dim, -1).transpose(0, 1).flatten(1, 2)

        self.copy(
            attn_proj_weight, 
            attn.linear_qkv.weight,
            param_type=ParamType.QKV_W,
        )
        self.copy(
            hf_attn.proj.weight,
            attn.linear_proj.weight,
            param_type=ParamType.ROW
        )
        self.copy(
            hf_attn.proj.bias,
            attn.linear_proj.bias,
        )

        # Copy bias
        if self.args.add_qkv_bias:
            if self.dryrun:
                attn_proj_bias = attn.linear_qkv.bias
            else:
                attn_proj_bias = self.load_tensor(
                    hf_attn.qkv.bias
                ).view(num_querys_per_group + 2, num_query_groups, dim, -1).transpose(0, 1).flatten(1, 2)
            self.copy(
                attn_proj_bias, 
                attn.linear_qkv.bias,
                param_type=ParamType.QKV_B,
            )

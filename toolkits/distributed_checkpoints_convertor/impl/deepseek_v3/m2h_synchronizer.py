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
import torch

from transformers import AutoConfig

from general.m2h_synchronizer import MG2HFSynchronizer as _MG2HFSynchronizer
from general.synchronizer import ParamType
from deepseek_v3.patch import add_mtp_layers

class MG2HFSynchronizer(_MG2HFSynchronizer):

    def __init__(self, load_dir, model_provider_func=None):
        super().__init__(
            load_dir, 
            model_provider_func=model_provider_func
        )
        if self.args.mtp_num_layers is not None:
            config = AutoConfig.from_pretrained(self.load_dir, trust_remote_code=True)
            self.hf_mtp_offset = config.num_hidden_layers
            self._hfmodel = add_mtp_layers(self._hfmodel, config, self.args.mtp_num_layers).to(config.torch_dtype)
            self.build_hf_mapping()
            self._merge_type: torch.Tensor = torch.zeros([self.hf_size], dtype=torch.int, device=self.device)

    def sync_params(self):
        super().sync_params()
        if self._mgmodel.mtp_process:
            for mtp_layer_id in range(self.args.mtp_num_layers):
                hf_mtp_layer_id = self.hf_mtp_offset + mtp_layer_id
                self.copy(
                    self._mgmodel.embedding.word_embeddings.weight, 
                    self._hfmodel.model.layers[hf_mtp_layer_id].embed_tokens.weight, 
                    param_type=ParamType.COLUMN
                )
                self.set_mtp_layer_state(
                    self._mgmodel.mtp.layers[mtp_layer_id],
                    self._hfmodel.model.layers[hf_mtp_layer_id]
                )
                self.copy(
                    self._mgmodel.output_layer.weight, 
                    self._hfmodel.model.layers[hf_mtp_layer_id].shared_head.head.weight, 
                    param_type=ParamType.COLUMN
                )            
    
    def set_mtp_layer_state(self, mtp, hf_mtp):
        self.copy(mtp.enorm.weight, hf_mtp.enorm.weight)
        self.copy(mtp.hnorm.weight, hf_mtp.hnorm.weight)
        self.copy(mtp.eh_proj.weight, hf_mtp.eh_proj.weight, param_type=ParamType.COLUMN)
        self.copy(mtp.final_layernorm.weight, hf_mtp.shared_head.norm.weight)
        self.set_layer_state(mtp.transformer_layer, hf_mtp)
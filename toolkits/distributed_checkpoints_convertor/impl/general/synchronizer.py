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
import logging

from typing import *

from enum import Enum
from abc import ABC, abstractmethod
from torch import distributed as dist
import transformers
from transformers import AutoConfig
from accelerate import init_empty_weights

from megatron.training import get_args
from megatron.core import parallel_state as mpu


class ParamType(Enum):
    NULL = 0
    COLUMN = 1 # along axis 0
    ROW = 2 # along axis 1
    QKV_W = 3 # qkv_proj
    QKV_B = 4
    UNIQUE = 5
    GATE_UP = 6
    MOE_COLUMN = 7
    MOE_ROW = 8
    MOE_GATE_UP = 9
    # generalized gate_up with multi linear with different output size
    MERGED_LINEAR = 10
    QGKV_W = 11

class BaseSynchronizer(ABC):
    def __init__(
            self, 
            load_dir, 
            model_provider_func = None
        ):
        """The base class of a parameter synchronizer.

        Args:
            load_dir (str): The path of a pretrained huggingface checkpiont.
            use_gpu (bool, optional): If sync with CUDA device. Defaults to False.
        """
        self.args = get_args()
        self.debug = self.args.debug
        self.dryrun = self.args.dryrun
        self.load_dir = load_dir
        self.device = torch.device(torch.cuda.current_device()) if self.args.use_gpu else 'cpu'

        self.tp_rank, self.tp_size = mpu.get_tensor_model_parallel_rank(), self.args.tensor_model_parallel_size
        self.pp_rank, self.pp_size = mpu.get_pipeline_model_parallel_rank(), self.args.pipeline_model_parallel_size
        self.ep_rank, self.ep_size = mpu.get_expert_model_parallel_rank(), self.args.expert_model_parallel_size
        self.etp_rank, self.etp_size = mpu.get_expert_tensor_parallel_rank(), self.args.expert_tensor_parallel_size
        self.dp_rank, self.edp_rank = mpu.get_data_parallel_rank(), mpu.get_expert_data_parallel_rank()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        pre_process = True if self.pp_rank == 0 else False
        post_process = True if self.pp_rank == self.pp_size - 1 else False
        if model_provider_func is None:
            # NOTE: try to import default model provider
            from pretrain_gpt import model_provider
            model_provider_func = model_provider
        self._mgmodel = model_provider_func(pre_process, post_process)

        config = AutoConfig.from_pretrained(self.load_dir, trust_remote_code=True)
        with init_empty_weights(include_buffers=True):
            automodel_cls = getattr(transformers, self.args.auto_model)
            self._hfmodel = automodel_cls.from_config(config, trust_remote_code=True, torch_dtype=config.torch_dtype)

        self.build_hf_mapping()

    def build_hf_mapping(self):
        # NOTE: two or more keys may point to the same tensor and we need to deduplicate
        state_dict = self._hfmodel.state_dict(keep_vars=True)
        # NOTE: find unique tensor and assign id
        self._hf_params_to_key = {v: k for k, v in state_dict.items()} 
        keys_to_id = {k: i for i, k in enumerate(sorted(self._hf_params_to_key.values()))}
        self._hf_params_to_id = {k: keys_to_id[v] for k, v in self._hf_params_to_key.items()}
        self._hf_params_key_to_id = {k: self._hf_params_to_id[v] for k, v in state_dict.items()}
        self._id_to_hf_params_key = {v: k for k, v in keys_to_id.items()}

        assert all(idx in self._hf_params_to_id.values() for idx in range(self.hf_size)), \
            f"Unexpected hf mapping, the desired range is [0, {self.hf_size}) but got [{min(self._hf_params_to_id.values())}, {max(self._hf_params_to_id.values())})"

    @property
    def hf_size(self):
        return len(self._hf_params_to_id)

    @abstractmethod
    def _copy_impl(self, src_tensor, dst_tensor, **kwargs):
        ...
    
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
            self.set_postprocess_state(mg_model=mg_model, hf_model=hf_model)

        for mg_layer_id, hf_layer_id in self._build_pipeline_parallel_mapping().items():
            if self.tp_rank == 0 and self.ep_rank == 0 and self.etp_rank == 0:
                logging.info(f"Converting layer {hf_layer_id}")
            layer = mg_model.decoder.layers[mg_layer_id]
            hf_layer = hf_model.model.layers[hf_layer_id]
            self.set_layer_state(layer, hf_layer)

    @abstractmethod
    def set_preprocess_state(self, mg_model, hf_model):
        ...

    @abstractmethod
    def set_postprocess_state(self, mg_model, hf_model):
        ...

    @abstractmethod
    def check_and_save(self, output_dir):
        ...

    def copy(self, src_tensor, dst_tensor, **kwargs):
        return self._copy_impl(src_tensor, dst_tensor, **kwargs)
    
    def _build_pipeline_parallel_mapping(self) -> Dict[int, int]:
        remained_num_layers = self.args.num_layers
        remained_stages = self.pp_size
        pp_layers_per_stage = []
        if self.args.decoder_first_pipeline_num_layers is not None:
            pp_layers_per_stage.append(self.args.decoder_first_pipeline_num_layers)
            remained_num_layers -= self.args.decoder_first_pipeline_num_layers
            remained_stages -= 1
        
        if self.args.decoder_last_pipeline_num_layers is not None:
            remained_num_layers -= self.args.decoder_last_pipeline_num_layers
            remained_stages -= 1
            assert remained_stages >= 0 and remained_num_layers >=0, 'Uneven PP: first + last > total, error'
            if remained_stages == 0:
                assert remained_num_layers ==0, 'Uneven PP: Unexpected Layers'
            else:
                assert remained_num_layers > 0 and remained_num_layers % remained_stages == 0, 'Invalid Uneven PP setting'
        
        if remained_stages > 0:
            pp_layers_per_stage.extend([remained_num_layers // remained_stages] * remained_stages)
        
        if self.args.decoder_last_pipeline_num_layers is not None:
            pp_layers_per_stage.append(self.args.decoder_last_pipeline_num_layers)

        pp_mapping = {
            i: v for i, v in enumerate(
                range(sum(pp_layers_per_stage[:self.pp_rank]), sum(pp_layers_per_stage[:self.pp_rank + 1]))
            )
        }
        return pp_mapping
    
    def _build_expert_parallel_mapping(self) -> Dict[int, int]:
        num_experts = self.args.num_experts
        if num_experts % self.ep_size != 0:
            raise ValueError()
        num_experts_per_rank = num_experts // self.ep_size
        expert_mapping = {
            i: v for i, v in enumerate(
                range(num_experts_per_rank * self.ep_rank, num_experts_per_rank * (self.ep_rank + 1))
            )
        }
        return expert_mapping
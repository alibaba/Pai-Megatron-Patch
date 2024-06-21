# Copyright (c) 2024 Alibaba PAI and Nvidia Megatron-LM Team.
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

from dataclasses import dataclass
from megatron.core.transformer import TransformerConfig

@dataclass
class Qwen2TransformerConfig(TransformerConfig):

    transformer_impl: str = 'transformer_engine'

    moe_ffn_hidden_size: int = None

    shared_moe_ffn_hidden_size: int = None

    enable_shared_expert: bool = False

    num_shared_experts: int = None

    moe_layer_freq: int = None

    rotary_base: int = None

    rotary_scaling_factor: int = None

    max_position_embeddings: int = None

    moe_aux_loss_coeff: float = 0.0
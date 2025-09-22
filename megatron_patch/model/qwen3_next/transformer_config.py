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
import torch
from typing import List
from dataclasses import dataclass, field
from megatron.core.transformer import TransformerConfig
from transformers import AutoConfig

@dataclass
class Qwen3NextTransformerConfig(TransformerConfig):

    head_k_dim: int = 128
    head_v_dim: int = 128
    num_k_heads: int = 16
    num_v_heads: int = 32



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

import torch
from torch import nn
from torch.nn import functional as F


class _GLUBaseModule(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.activation_fn = activation_fn

    def forward(self, x):
        # dim=-1 breaks in jit for pt<1.10
        x1, x2 = x.chunk(2, dim=(x.ndim - 1))
        return x1 * self.activation_fn(x2)


class LiGLU(_GLUBaseModule):
    def __init__(self):
        super().__init__(nn.Identity())


class GEGLU(_GLUBaseModule):
    def __init__(self):
        super().__init__(F.gelu)


class ReGLU(_GLUBaseModule):
    def __init__(self):
        super().__init__(F.relu)


class SwiGLU(_GLUBaseModule):
    def __init__(self):
        super().__init__(F.silu)


liglu = torch.jit.script(LiGLU())
geglu = torch.jit.script(GEGLU())
reglu = torch.jit.script(ReGLU())
swiglu = torch.jit.script(SwiGLU())


GLU_ACTIVATIONS = {
    "geglu": geglu,
    "liglu": liglu,
    "reglu": reglu,
    "swiglu": swiglu,
}

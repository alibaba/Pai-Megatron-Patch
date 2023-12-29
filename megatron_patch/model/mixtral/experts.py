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
import copy

class Experts(torch.nn.Module):
    """ A module that holds multiple expert models and processes input chunks in parallel """
    def __init__(self, expert, num_local_experts=1, expert_group_name=None):
        """
        Initialize the Experts module.

        Args:
            expert: A PyTorch model representing a single expert. This model could
                    be shared among multiple experts if they are identical.
            num_local_experts: The number of experts that will process input chunks
                               in parallel.
        """
        super(Experts, self).__init__()

        self.megatron_experts = torch.nn.ModuleList(
            [copy.deepcopy(expert) for i in range(num_local_experts)])
        self.num_local_experts = num_local_experts

        # TODO: revisit allreduce for moe.gate...
        for expert in self.megatron_experts:
            # TODO: Create param groups to handle expert + data case (e.g. param.group = moe_group)
            for name, param in expert.named_parameters():
                param.allreduce = False
                param.group_name = expert_group_name

    def forward(self, inputs):
        """
        Forward pass for the Experts module.

        Args:
            inputs: The input tensor to be processed by the experts.

        Returns:
            Tensor: The concatenated output tensor from all experts.
        """
         
        # Check if the inputs can be evenly divided by the number of experts
        if inputs.size(1) % self.num_local_experts != 0:
            raise ValueError("The number of input features is not evenly divisible by the number of experts.")

        chunks = inputs.chunk(self.num_local_experts, dim=1)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.megatron_experts):
            out = expert(chunk)
            if type(out) is tuple:
                out = out[0]  # Ignore the bias term for now
            expert_outputs += [out]

        expert_output = torch.cat(expert_outputs, dim=1)
        return expert_output

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

from .router import MOELayer, Router
from .experts import Experts
import typing
from megatron import get_args
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)

class MoE(torch.nn.Module):
    def __init__(self,
                 hidden_size,
                 expert,
                 num_experts=1,
                 ep_size=1,
                 k=1,
                 capacity_factor=1.,
                 eval_capacity_factor=1.,
                 min_capacity=4,
                 use_residual=False,
                 noisy_gate_policy: typing.Optional[str] = None,
                 drop_tokens: bool = True,
                 use_rts=True,
                 use_tutel: bool = False,
                 expert_tensor_parallelism: bool = False,
                 moe_layer_index: int = None):
        """Initialize an MoE layer.

        Arguments:
            hidden_size (int): the hidden dimension of the model, importantly this is also the input and output dimension.
            expert (torch.nn.Module): the torch module that defines the expert (e.g., MLP, torch.linear).
            num_experts (int, optional): default=1, the total number of experts per layer.
            ep_size (int, optional): default=1, number of ranks in the expert parallel world or group.
            k (int, optional): default=1, top-k gating value, only supports k=1 or k=2.
            capacity_factor (float, optional): default=1.0, the capacity of the expert at training time.
            eval_capacity_factor (float, optional): default=1.0, the capacity of the expert at eval time.
            min_capacity (int, optional): default=4, the minimum capacity per expert regardless of the capacity_factor.
            use_residual (bool, optional): default=False, make this MoE layer a Residual MoE (https://arxiv.org/abs/2201.05596) layer.
            noisy_gate_policy (str, optional): default=None, noisy gate policy, valid options are 'Jitter', 'RSample' or 'None'.
            drop_tokens (bool, optional): default=True, whether to drop tokens - (setting to False is equivalent to infinite capacity).
            use_rts (bool, optional): default=True, whether to use Random Token Selection.
            use_tutel (bool, optional): default=False, whether to use Tutel optimizations (if installed).
            expert_tensor_parallelism (bool, optional): default=False, whether to use tensor parallelism for experts
        """

        super(MoE, self).__init__()

        self.use_residual = use_residual
        self.expert_tensor_parallelism = expert_tensor_parallelism
        assert num_experts % ep_size == 0, f"Number of experts ({num_experts}) should be divisible by expert parallel size ({ep_size})"
        self.ep_size = ep_size
        self.expert_group_name = f"ep_size_{self.ep_size}"
        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.ep_size
        self.moe_layer_index = moe_layer_index

        experts = Experts(expert, self.num_local_experts, self.expert_group_name)
        self.megatron_moe = MOELayer(Router(hidden_size,
                                            num_experts,
                                            k,
                                            capacity_factor,
                                            eval_capacity_factor,
                                            min_capacity,
                                            noisy_gate_policy,
                                            drop_tokens,
                                            use_rts),
                                      experts,
                                      self.expert_group_name,
                                      self.ep_size,
                                      self.num_local_experts,
                                      use_tutel=use_tutel,
                                      expert_tensor_parallelism=expert_tensor_parallelism)
        if self.use_residual:
            self.mlp = expert
            # coefficient is used for weighted sum of the output of expert and mlp
            self.coefficient = torch.nn.Linear(hidden_size, 2)

    def forward(self, hidden_states, used_token=None):
        """ MoE forward

        Arguments:
            hidden_states (Tensor): input to the layer
            used_token (Tensor, optional): default: None, mask only used tokens

        Returns:
            A tuple including output, gate loss, and expert count.

            * output (Tensor): output of the model

            * mlp_bias (Tensor): placehoder, no effect
        """
        args = get_args()
        
        # Gathering hidden_states for expert tensor parallel.
        if args.expert_tensor_parallelism and args.sequence_parallel:
            hidden_states = \
                gather_from_sequence_parallel_region(hidden_states, tensor_parallel_output_grad=False)

        output = self.megatron_moe(hidden_states, used_token)
        
        if self.use_residual:
            # Residual MoE
            output_mlp = self.mlp(hidden_states)
            if type(output_mlp) is tuple:
                output_mlp = output_mlp[0]  # Ignore the bias term for now
            coef = self.coefficient(hidden_states)
            coef = torch.nn.functional.softmax(coef, dim=-1)
            output = output * coef[..., 0:1] + output_mlp * coef[..., 1:]

        # Reduce hidden_states after expert tensor parallel
        if args.expert_tensor_parallelism:
            if args.sequence_parallel:
                output = reduce_scatter_to_sequence_parallel_region(output)
            else:
                output = reduce_from_tensor_model_parallel_region(output)

        mlp_bias = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        return output, mlp_bias

    def get_moe_layer_index(self):
        return self.moe_layer_index

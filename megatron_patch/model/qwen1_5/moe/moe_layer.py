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

from abc import ABC, abstractmethod
import torch

from megatron.core import parallel_state
from megatron.core.transformer.module import MegatronModule

from .experts import GroupedMLP, SequentialMLP
from .router import TopKRouter
from .token_dispatcher import MoEDroplessTokenDispatcher
from ..transformer_config import TransformerConfig
from ..transformer.mlp import MLPSubmodules

class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(self, config: TransformerConfig):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
        assert self.config.num_moe_experts % self.expert_parallel_size == 0
        self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )
        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        self.router = None
        self.experts = None
        self.token_dispatcher = None

    @abstractmethod
    def forward(self, hidden_states):
        pass


class MoELayer(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(self, config: TransformerConfig, submodules: MLPSubmodules = None):
        self.submodules = submodules
        super(MoELayer, self).__init__(config=config)
        self.router = TopKRouter(
            self.num_local_experts, self.local_expert_indices, config=self.config
        )
        if self.config.moe_grouped_gemm:
            self.experts = GroupedMLP(self.num_local_experts, self.config)
        else:
            assert isinstance(self.submodules, MLPSubmodules)
            self.experts = SequentialMLP(self.num_local_experts, self.config, self.submodules)
        self.token_dispatcher = MoEDroplessTokenDispatcher(
            self.num_local_experts, self.local_expert_indices, config=self.config
        )

    def forward(self, hidden_states: torch.Tensor):
        """
        Forward pass for the MoE layer.

        The method routes input tokens to the appropriate expert networks,
        processes the tokens with the experts, and then combines the outputs.

        Args:
            hidden_states (torch.Tensor): The input tensor containing the hidden states
            from the previous layer of the transformer model.This tensor is expected to 
            have a shape compatible with the expectations of the MoE layer, typically
            [batch_size, sequence_length, hidden_size].

        Returns:
            Tupletorch.Tensor, torch.Tensor: A tuple containing two elements:
                - The first element is the output tensor after processing by the MoE layer.
                  It has the same shape as the input hidden_states.
                - The second element is the bias introduced by the MLP experts, which may
                need to be accounted for in subsequent layers or loss calculations.
        """
        # process MoE
        scores, indices = self.router(hidden_states)
        (
            dispatched_input,
            tokens_per_expert,
            scores,
            indices,
            global_local_map,
        ) = self.token_dispatcher.token_permutation(hidden_states, scores, indices)
        expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
        output, mlp_bias = self.token_dispatcher.token_unpermutation(
            expert_output, scores, indices, global_local_map, mlp_bias
        )
        return output, mlp_bias

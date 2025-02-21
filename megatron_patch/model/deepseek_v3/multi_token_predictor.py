# Copyright (c) 2025 Alibaba PAI and Nvidia Megatron-LM Team.
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
# Some MPT Implementation copy from: https://github.com/FlagOpen/FlagScale

import torch
from torch import Tensor

from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_block import (
    TransformerBlock,
    TransformerBlockSubmodules,
)

try:
    from megatron.core.extensions.transformer_engine import (
        TENorm,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


class DeepSeekMultiTokenPredictorLayer(MegatronModule):
    """Multi Token Prediction Layer of DeepSeek V3

    Args:
        config (TransformerConfig): config object with all necessary configs for TransformerBlock
    """

    def __init__(
            self,
            config: TransformerConfig,
            transformer_layer_spec: ModuleSpec,
    ):
        super().__init__(config=config)

        self.config = config

        if HAVE_TE:
            self.norm1 = TENorm(config, config.hidden_size, config.layernorm_epsilon)
            self.norm2 = TENorm(config, config.hidden_size, config.layernorm_epsilon)
        else:
            self.norm1 = torch.nn.RMSNorm(normalized_shape=config.hidden_size, eps=config.layernorm_epsilon)
            self.norm2 = torch.nn.RMSNorm(normalized_shape=config.hidden_size, eps=config.layernorm_epsilon)

        self.linear_proj = torch.nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

        # the transformer block, fork from main model or use a user-defined transformer layer spec?
        if isinstance(transformer_layer_spec, TransformerBlockSubmodules):
            transformer_layer_spec = transformer_layer_spec.layer_specs[-1]
        self.decoder = TransformerBlock(
            config=config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=True,
        )

    def forward(
            self,
            decoder_input: Tensor,
            attention_mask: Tensor,
            pre_hidden_states: Tensor,

    ) -> Tensor:
        """Forward pass of the multi token prediction layer.
        """
        assert decoder_input is not None, "Input ids need to be embedded before mtp predictor"

        # two RMSNorm
        decoder_input = self.norm1(decoder_input)
        pre_hidden_states = self.norm2(pre_hidden_states)
        # concat
        hidden_states = torch.cat([pre_hidden_states, decoder_input], dim=-1)
        # linear projection
        hidden_states = self.linear_proj(hidden_states)
        # transformer block
        hidden_states = self.decoder(hidden_states, attention_mask)

        return hidden_states


class DeepSeekMultiTokenPredictor(MegatronModule):
    """Multi Token Predictor of DeepSeek V3

    Args:
        config (TransformerConfig): config object with all necessary configs for TransformerBlock
    """

    def __init__(
            self,
            config: TransformerConfig,
            transformer_layer_spec: ModuleSpec,
    ):
        super().__init__(config=config)

        self.config = config
        self.num_mtp_predictor = config.num_mtp_predictor

        self.mtp_modules = torch.nn.ModuleList([
            DeepSeekMultiTokenPredictorLayer(
                config=self.config,
                transformer_layer_spec=transformer_layer_spec,
            ) for i in range(self.num_mtp_predictor)
        ])

    def forward(
            self,
            decoder_input: Tensor,
            attention_mask: Tensor,
            pre_hidden_states: Tensor,
    ) -> Tensor:
        """Forward pass of the multi token prediction module.
        """

        hidden_states_mtps = []
        for i in range(self.num_mtp_predictor):
            decoder_input, _ = roll_tensor(decoder_input, dims=0)
            hidden_states = self.mtp_modules[i](
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                pre_hidden_states=pre_hidden_states,
            )
            hidden_states_mtps.append(hidden_states)
            pre_hidden_states = hidden_states

        return hidden_states_mtps


def roll_tensor(tensor, dims=0):
    rolled_tensor = torch.roll(tensor, shifts=-1, dims=dims)
    index = [slice(None)] * rolled_tensor.ndim
    index[dims] = -1
    index = tuple(index)
    rolled_tensor[index] = 0
    return rolled_tensor, rolled_tensor.sum()
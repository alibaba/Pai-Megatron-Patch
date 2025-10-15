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
from functools import partial
import torch
import torch._dynamo

from megatron.core.enums import ModelType
from model_provider import model_provider as base_model_provider # Megatron-LM-250908/model_provider.py

from megatron.training.arguments import core_transformer_config_from_args
from megatron_patch.arguments import get_patch_args
from megatron_patch.data import train_valid_test_datasets_provider
from megatron.training import pretrain, print_rank_0

torch._dynamo.config.suppress_errors = True

from megatron.core.models.mamba import MambaModel
from megatron.training import print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from megatron_patch.model.qwen3_next.layer_specs import get_qwen3_next_layer_spec
from megatron_patch.model.qwen3_next.transformer_config import Qwen3NextTransformerConfig

def mamba_builder(args, pre_process, post_process, vp_stage=None, config=None):
    print_rank_0('building MAMBA model ...')
    if config is None:
        config = core_transformer_config_from_args(args, Qwen3NextTransformerConfig)
    assert args.use_legacy_models is False, "Mamba only supported in Mcore!"

    model = MambaModel(
        config=config,
        mamba_stack_spec=get_qwen3_next_layer_spec(args),
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        hybrid_attention_ratio=args.hybrid_attention_ratio,
        hybrid_mlp_ratio=args.hybrid_mlp_ratio,
        hybrid_override_pattern=args.hybrid_override_pattern,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
    )

    return model

model_provider = partial(base_model_provider, mamba_builder)


if __name__ == "__main__":
    from megatron_patch.template.helper import forward_step
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=get_patch_args,
    )
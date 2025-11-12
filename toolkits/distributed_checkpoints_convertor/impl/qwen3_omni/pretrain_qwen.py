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
"""Pretrain Qwen3-Omni."""
from copy import deepcopy
from typing import Union, Optional

import torch
import torch._dynamo

from transformers import AutoConfig

from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from megatron_patch.model.qwen3_vl.layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_qwen3vl_vision_model_spec,
    get_mlp_module_spec
)
from megatron_patch.model.qwen3_omni.model import Qwen3OmniModel

torch._dynamo.config.suppress_errors = True
from megatron_patch.model.qwen3_omni.transformer_config import (
    Qwen3OmniTransformerConfig,
    get_vision_model_config,
    get_vision_projection_config
)


def model_provider(
    pre_process=True, post_process=True, add_encoder=True, add_decoder=True, vp_stage: Optional[int] = None
) -> Union[Qwen3OmniModel]:
    args = get_args()
    print_rank_0("start building qwen3-omni model ...")

    # Config of vit, llm and projector
    config = core_transformer_config_from_args(args, Qwen3OmniTransformerConfig)
    use_te = args.transformer_impl == "transformer_engine"
    if not use_te:
        raise NotImplementedError("The Qwen3-Omni model is only implemented with TransformerEngine!")
    
    if args.rotary_seq_len_interpolation_factor is not None or args.rotary_seq_len_interpolation_factor != 1:
        print_rank_0('Multimodal RoPE currently not support RoPE interpolation, set to None...')
        args.rotary_seq_len_interpolation_factor = None

    vision_config = get_vision_model_config(args, deepcopy(config))
    vision_config.pipeline_model_parallel_size = 1
    vision_config.num_layers_in_first_pipeline_stage = None
    vision_projector_config = get_vision_projection_config(deepcopy(config), vision_config.hidden_size, vision_config.spatial_merge_size)
    
    print_rank_0("building Qwen3-Omni model in TE...")
    # Layer Specs of vit, llm and projector
    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=args.num_experts,
        moe_grouped_gemm=args.moe_grouped_gemm,
        qk_layernorm=args.qk_layernorm
    )
    vision_model_spec = get_qwen3vl_vision_model_spec()
    vision_projector_spec = get_mlp_module_spec(add_norm=False).submodules

    qwen3_omni_hf_config = AutoConfig.from_pretrained(args.load_dir)
    audio_hf_config = qwen3_omni_hf_config.thinker_config.audio_config

    model = Qwen3OmniModel(
        language_transformer_config=config,
        language_transformer_layer_spec=transformer_layer_spec,
        language_vocab_size=args.padded_vocab_size,
        language_max_sequence_length=args.max_position_embeddings,

        vision_transformer_config=vision_config,
        vision_transformer_layer_spec=vision_model_spec,
        drop_vision_class_token=False, # NOTE: no class token to drop?

        vision_projection_config=vision_projector_config,
        vision_projection_layer_spec=vision_projector_spec, 
        vision_projection_type='mlp',

        audio_transformer_config=audio_hf_config,

        allow_missing_vision_projection_checkpoint= False, # TODO: may parameterized

        language_position_embedding_type=args.position_embedding_type,
        language_rotary_percent=args.rotary_percent,
        language_rotary_base=args.rotary_base,
        
        pre_process=pre_process,
        post_process=post_process,
        add_decoder=add_decoder,
        add_encoder=add_encoder,

        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        language_share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        vp_stage=vp_stage
    )

    model.freeze(
        freeze_language_model=getattr(args, 'freeze_LM', False), 
        freeze_vision_model=getattr(args, 'freeze_ViT', False), 
        freeze_vision_projection=False
    )

    return model
import torch
from dataclasses import dataclass
from megatron.core.transformer import MLATransformerConfig

import dataclasses
import torch.nn.functional as F
from megatron.core.transformer import TransformerConfig
def core_transformer_config_from_args(args, config_class=None):
    # Config class.
    config_class = config_class or TransformerConfig

    # Translate args to core transformer configuration
    kw_args = {}
    for f in dataclasses.fields(config_class):
        if hasattr(args, f.name):
            kw_args[f.name] = getattr(args, f.name)
    kw_args['persist_layer_norm'] = not args.no_persist_layer_norm
    kw_args['layernorm_zero_centered_gamma'] = args.apply_layernorm_1p
    kw_args['layernorm_epsilon'] = args.norm_epsilon
    kw_args['deallocate_pipeline_outputs'] = True
    kw_args['pipeline_dtype'] = args.params_dtype
    kw_args['batch_p2p_comm'] = not args.overlap_p2p_comm
    kw_args['num_moe_experts'] = args.num_experts
    kw_args['rotary_interleaved'] = args.rotary_interleaved
    kw_args['first_pipeline_num_layers'] = args.decoder_first_pipeline_num_layers
    kw_args['last_pipeline_num_layers'] = args.decoder_last_pipeline_num_layers
    if args.swiglu:
        kw_args['activation_func'] = F.silu
        kw_args['gated_linear_unit'] = True
        kw_args['bias_activation_fusion'] = args.bias_swiglu_fusion
    else:
        kw_args['bias_activation_fusion'] = args.bias_gelu_fusion

    if args.init_method_xavier_uniform:
        kw_args['init_method'] = torch.nn.init.xavier_uniform_
        kw_args['scaled_init_method'] = torch.nn.init.xavier_uniform_
    if args.group_query_attention:
        kw_args['num_query_groups'] = args.num_query_groups
    else:
        kw_args['num_query_groups'] = None
    kw_args['config_logger_dir'] = args.config_logger_dir

    # Return config.
    return config_class(**kw_args)

@dataclass
class DeepSeekV2TransformerConfig(MLATransformerConfig):

    moe_ffn_hidden_size: int = None

    moe_layer_freq: int = None

    original_max_position_embeddings: int = 4096
    """Maximum position embeddings for the original model, used by yarn."""
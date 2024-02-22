# Copyright (c) 2023 Alibaba PAI Team.
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
import argparse
import dataclasses
import torch.nn.functional as F


def core_transformer_config_from_args(args, TransformerConfig):

    # Translate args to core transformer configuration
    kw_args = {}
    for f in dataclasses.fields(TransformerConfig):
        if hasattr(args, f.name):
            kw_args[f.name] = getattr(args, f.name)
    kw_args['persist_layer_norm'] = not args.no_persist_layer_norm
    kw_args['layernorm_zero_centered_gamma'] = args.apply_layernorm_1p
    kw_args['layernorm_epsilon'] = args.norm_epsilon
    kw_args['deallocate_pipeline_outputs'] = True
    kw_args['pipeline_dtype'] = args.params_dtype
    kw_args['batch_p2p_comm'] = not args.overlap_p2p_comm
    kw_args['num_moe_experts'] = args.num_experts
    if args.swiglu:
        kw_args['activation_func'] = F.silu
        kw_args['gated_linear_unit'] = True
        kw_args['bias_activation_fusion'] = args.bias_swiglu_fusion
    else:
        kw_args['bias_activation_fusion'] = args.bias_gelu_fusion
    if args.squared_relu:
        assert not args.swiglu
        def squared_relu(x):
            return torch.pow(F.relu(x), 2)
        kw_args['activation_func'] = squared_relu
    if args.init_method_xavier_uniform:
        kw_args['init_method'] = torch.nn.init.xavier_uniform_
        kw_args['scaled_init_method'] = torch.nn.init.xavier_uniform_
    if args.group_query_attention:
        kw_args['num_query_groups'] = args.num_query_groups
    else:
        kw_args['num_query_groups'] = None

    # Return Transformer config.
    return TransformerConfig(**kw_args)

def get_patch_args(parser):
    group = parser.add_argument_group(title='patch')

    for action in vars(group)['_actions']:
        if isinstance(action, argparse._StoreAction):
            if '--tokenizer-type' in action.option_strings:
                action.default = "NullTokenizer"

    for action in vars(group)['_actions']:
        if isinstance(action, argparse._StoreAction):
            if '--vocab-size' in action.option_strings:
                action.default = -1

    for action in vars(group)['_actions']:
        if isinstance(action, argparse._StoreAction):
            if '--position-embedding-type' in action.option_strings:
                action.choices.append('none')

    group.add_argument('--local-rank',
                       type=int,
                       default=None,
                       help='local rank passed from distributed launcher')

    group.add_argument('--n-head-kv',
                       type=int,
                       default=None,
                       help='n-head-kv')

    group.add_argument('--transformer-type',
                       type=str,
                       default='megatron',
                       help='transformer-type')

    group.add_argument('--max-padding-length',
                       type=int,
                       default=None,
                       help='max-padding-length')

    group.add_argument('--dataset',
                       type=str,
                       default=None,
                       help='dataset')

    group.add_argument('--pretrained-checkpoint',
                       type=str,
                       default=None,
                       help='Pretrained checkpoint used for finetunning.')

    group.add_argument('--epochs',
                       type=int,
                       default=None,
                       help='Number of finetunning epochs. Zero results in '
                       'evaluation only.')

    group.add_argument('--intermediate-size',
                       type=int,
                       default=None,
                       help='--intermediate-size')

    group.add_argument('--extra-vocab-size',
                       type=int,
                       default=0,
                       help='--extra-vocab-size')

    group.add_argument('--keep-last',
                       action='store_true',
                       help='Keep the last batch (maybe incomplete) in'
                       'the data loader')

    group.add_argument('--data-dir',
                       default=None,
                       help='data-dir')

    group.add_argument('--train-data',
                       nargs='+',
                       default=None,
                       help='Whitespace separated paths or corpora names '
                       'for training.')

    group.add_argument('--valid-data',
                       nargs='+',
                       default=None,
                       help='path(s) to the validation data.')

    group.add_argument('--patch-tokenizer-type',
                       type=str,
                       help='patch-tokenizer-type')

    group.add_argument('--use-alibi-mask',
                       action='store_true',
                       help='use alibi mask for baichuan model')

    group.add_argument('--use-normhead',
                       action='store_true',
                       help='use-normhead')

    group.add_argument('--glu-activation',
                       type=str,
                       help='GLU activations to use.')

    group.add_argument('--attention-head-type',
                       type=str,
                       default=None,
                       choices=['multihead', 'multiquery'],
                       help='Type of attention heads. `multihead` is the standard multi-head attention.'
                       '`multiquery` shares the values and keys across attention heads')

    group.add_argument('--transformer-timers',
                       action='store_true',
                       help="If set, activate the timers within the transformer layers."
                        "Only for debugging, as this slows down the model.")

    group.add_argument('--text-generate-input-file',
                       type=str,
                       default='')

    group.add_argument('--text-generate-output-file',
                       type=str,
                       default='')

    group.add_argument('--text-generate-gt-file',
                       type=str,
                       default='')

    group.add_argument('--time',
                       action='store_true',
                       help='measure end to end text generation average time')

    group.add_argument('--eval-dev',
                       action='store_true')

    group.add_argument('--input-len',
                       type=int,
                       default=1,
                       help='input lenth for measure end to end text generation average time')

    group.add_argument('--generation-length',
                       type=int,
                       default=None,
                       help='generation-seq-len')

    group.add_argument('--top-p',
                       type=float,
                       default=0.0,
                       help='Top p sampling.')

    group.add_argument('--top-k',
                       type=int,
                       default=0,
                       help='Top k sampling.')

    group.add_argument('--out-seq-length',
                       type=int,
                       default=1024,
                       help='Size of the output generated text.')

    group.add_argument('--temperature',
                       type=float,
                       default=1.0,
                       help='Sampling temperature.')

    group.add_argument('--repetition_penalty',
                       type=float,
                       default=1.1,
                       help='Repetition_penalty.')

    group.add_argument('--embed-layernorm',
                       action='store_true',
                       help='use layernorm for embedding')

    group.add_argument('--repetition-penalty',
                       type=float,
                       default=1.2,
                       help='Repetition_penalty.')


    group.add_argument('--source-seq-len',
                       type=int,
                       default=None,
                       help='source-seq-len')

    group.add_argument('--target-seq-len',
                       type=int,
                       default=None,
                       help='target-seq-len')

    group.add_argument('--position-encoding-2d',
                       action='store_true',
                       help='position-encoding-2d')

    group.add_argument('--z-loss-weight',
                       type=float,
                       default=0.0,
                       help='the max-z weight for baichuan2')

    group.add_argument('--data-impl', type=str, default='mmap',
                       choices=['mmap', 'infer'],
                       help='Implementation of indexed datasets.')

    group.add_argument('--use-llama2-rotary-position-embeddings', action='store_true',
                       help='Use llama2 rotary positional embeddings or not. '
                       'Deprecated: use --position-embedding-type')

    group.add_argument('--use-mistral-rotary-position-embeddings', action='store_true',
                       help='Use llama2 rotary positional embeddings or not. '
                       'Deprecated: use --position-embedding-type')

    group.add_argument('--mm-use-im-start-end',
                       action='store_true')

    group.add_argument('--mm-use-im-patch-token',
                       action='store_true')

    group.add_argument('--tune-mm-mlp-adapter',
                       action='store_true')

    group.add_argument('--freeze-clip-vision-tower',
                       action='store_true')

    group.add_argument('--freeze-llm',
                       action='store_true')

    group.add_argument('--image-folder',
                       type=str,
                       default='')

    group.add_argument('--mm-vision-select-layer',
                       type=int,
                       default=None)

    group.add_argument('--vision-tower',
                       type=str,
                       default='')

    group.add_argument('--image-aspect-ratio',
                       type=str,
                       default='square')

    group.add_argument('--version',
                       type=str,
                       default='plain')

    group.add_argument('--mm-projector-type',
                       type=str,
                       default=None)

    group.add_argument('--image-size',
                       type=int,
                       default=None,
                       help='image-size')

    group.add_argument('--patch-size',
                       type=int,
                       default=None,
                       help='patch-size')

    group.add_argument('--sliding-window', type=int, default=None)

    group.add_argument('--rotary-base', type=int, default=10000)

    group.add_argument('--rotary-scale-factor', type=int, default=1)

    group.add_argument('--cvcuda-image-processing',
                    action='store_true')

    group.add_argument('--expert-tensor-parallelism', action='store_true',
                       default=False,
                       help="use tensor parallelism for expert layers in MoE")

    group.add_argument('--expert-interval', type=int, default=2,
                       help='Use experts in every "expert-interval" layers')

    group.add_argument('--moe', action='store_true')

    group.add_argument('--moe-topk', type=int, default=1,
                       help='moe-topk')

    group.add_argument('--moe-expert-parallel-size', type=int, default=None,
                       help='Degree of the MoE expert parallelism. By default, '
                       'the size of this value will be automatically determined.')

    group.add_argument('--moe-train-capacity-factor', type=float, default=1.0,
                       help='The capacity of the MoE expert at training time')

    group.add_argument('--moe-eval-capacity-factor', type=float, default=1.0,
                       help='The capacity of the MoE expert at eval time.')

    group.add_argument('--moe-min-capacity', type=int, default=4,
                       help='The minimum capacity per MoE expert regardless of the capacity_factor.')

    group.add_argument('--moe-loss-coeff', type=float, default=0.01,
                       help='Scaling coefficient for adding MoE loss to model loss')

    group.add_argument('--use-tutel', action='store_true',
                       help='Use Tutel optimization for MoE')

    group.add_argument('--router-type', type=str, default='topk',
                       choices=['topk', 'expert_choice'],
                       help='Options for router type, support top1 & top2 and expert_choice')

    group.add_argument('--moe-input-feature-slicing', action='store_true',
                       help='Enable moe all2all performance optimization.')

    group.add_argument('--disable-bias-linear-fc', action='store_false',
                       help='Disable bias in the linear layers',
                       dest='add_bias_linear_fc')

    group.add_argument('--disable-bias-attn-fc', action='store_false',
                       help='Disable bias in the linear layers',
                       dest='add_bias_attn_fc')

    group.add_argument('--disable-parallel-output', action='store_false',
                       help='Disable parallel-output',
                       dest='enable_parallel_output')

    group.add_argument('--task-list', type=str, default="all", help='Either "all" or comma separated list of tasks.')

    group.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        help="Logging verbosity",
    )

    group.add_argument('--adaptive-seq-len',  default = False, action='store_true',
                       help='Should the sequence length be adapted to the batch during evaluation,'
                            ' if in fp16 the results will be slightly different due to numerical'
                            ' errors but greatly speed up evaluation.')

    group.add_argument('--eval-fp32', default=False, action='store_true', help='Should the evaluation run in fp32')

    group.add_argument('--num-fewshot', type=int, default=None,
                       help='num fewshot')

    return parser

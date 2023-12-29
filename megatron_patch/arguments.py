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

import argparse

def validate_moe_args(args, defaults={}):
    if args.num_experts is not None:
        args.moe = True
        if args.moe_expert_parallel_size is None:
            args.moe_expert_parallel_size = args.data_parallel_size
            if args.tensor_model_parallel_size > 0 and not args.expert_tensor_parallelism:
                # EP will use the span of DP*TP
                args.moe_expert_parallel_size *= args.tensor_model_parallel_size
        if args.rank == 0:
            print('Experts set to %s, expert parallel size set to %d'
                  % (str(args.num_experts), args.moe_expert_parallel_size))
    else:
        args.moe = False

def get_tasks_args(parser):
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
                       default=1,
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

    group.add_argument('--moe-topk', type=int, default=1,
                       help='moe-topk')

    group.add_argument('--moe-expert-parallel-size', type=int, default=None,
                       help='Degree of the MoE expert parallelism. By default, '
                       'the size of this value will be automatically determined.')

    group.add_argument('--disable-moe-token-dropping', action='store_false',
                       help='Disable MoE expert token dropping.',
                       dest='moe_token_dropping')

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

    return parser

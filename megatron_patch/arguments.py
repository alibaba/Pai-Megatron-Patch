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

    return parser

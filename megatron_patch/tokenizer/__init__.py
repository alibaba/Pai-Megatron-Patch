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

import os
from megatron import print_rank_0
from megatron.tokenizer.tokenizer import _vocab_size_with_padding

_GLOBAL_TOKENIZER = None

def build_tokenizer(args):
    """
    Initialize tokenizer.
    Args:
        args (argparse.Namespace): Arguments containing the type of tokenizer to be used.

    Returns:
        tokenizer object: Initialized tokenizer object based on the provided arguments.
    """
    if args.rank == 0:
        print('> building {} tokenizer ...'.format(args.patch_tokenizer_type))
    # Select and instantiate the tokenizer.
    if args.patch_tokenizer_type == 'JiebaBPETokenizer':
        from .jiebabpe_tokenizer import JiebaBPETokenizer
        tokenizer = JiebaBPETokenizer(args.patch_vocab_file)
        args.padded_vocab_size = _vocab_size_with_padding(
            tokenizer.vocab_size, args)
    elif args.patch_tokenizer_type == 'BloomTokenizerFromHF':
        from transformers import BloomTokenizerFast as BloomTokenizer
        if args.load is None:
            tokenizer = BloomTokenizer.from_pretrained('bigscience/bloom-560m')
        else:
            tokenizer = BloomTokenizer.from_pretrained(args.load)
        args.padded_vocab_size = 250880
    elif args.patch_tokenizer_type == 'ChatGLMTokenizerFromHF':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm-6b',
                                                  trust_remote_code=True)
        args.padded_vocab_size = 130528
    elif args.patch_tokenizer_type == 'GLM10BZHTokenizerFromHF':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('THUDM/glm-10b-chinese',
                                                  trust_remote_code=True)
        args.padded_vocab_size = 50048
    elif args.patch_tokenizer_type == 'IcetkGLM130BTokenizer':
        from .icetk_glm130b_tokenizer import _IceTokenizer
        tokenizer = _IceTokenizer()
        args.padded_vocab_size = 150528
    elif args.patch_tokenizer_type == 'OPTTokenizer':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.load,
            model_max_length=args.seq_length,
            padding_side='right',
            use_fast=False,
        )
        DEFAULT_PAD_TOKEN = '<pad>'
        DEFAULT_EOS_TOKEN = '</s>'
        DEFAULT_BOS_TOKEN = '<s>'
        DEFAULT_UNK_TOKEN = '<unk>'

        special_tokens_dict = dict()
        if not tokenizer.pad_token:
            special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN
        if not tokenizer.eos_token:
            special_tokens_dict['eos_token'] = DEFAULT_EOS_TOKEN
        if not tokenizer.bos_token:
            special_tokens_dict['bos_token'] = DEFAULT_BOS_TOKEN
        if not tokenizer.unk_token:
            special_tokens_dict['unk_token'] = DEFAULT_UNK_TOKEN
        tokenizer.add_special_tokens(special_tokens_dict)
        args.padded_vocab_size = tokenizer.vocab_size + args.extra_vocab_size

    elif args.patch_tokenizer_type == 'LLamaTokenizer-ziya':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.load, use_fast=False)
        args.padded_vocab_size = tokenizer.vocab_size + args.extra_vocab_size
        tokenizer.pad_token = tokenizer.eos_token

    elif args.patch_tokenizer_type == 'LLamaTokenizer':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.load,
            model_max_length=args.seq_length,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True
        )

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens(special_tokens_dict=dict(pad_token="<unk>"))

        tokenizer.eod = tokenizer.eos_token_id
        args.padded_vocab_size = tokenizer.vocab_size + args.extra_vocab_size

    elif args.patch_tokenizer_type == 'FalconTokenizer':
        from transformers import AutoTokenizer
        if args.load is None:
            tokenizer = AutoTokenizer.from_pretrained(
                'tiiuae/falcon-7b',
                model_max_length=args.seq_length,
                padding_side='right',
                use_fast=False,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                args.load,
                model_max_length=args.seq_length,
                padding_side='right',
                use_fast=False,
            )
        args.padded_vocab_size = tokenizer.vocab_size + args.extra_vocab_size
        tokenizer.pad_token = tokenizer.eos_token
    elif args.patch_tokenizer_type == 'BaichuanTokenizer':
        from .tokenization_baichuan import BaichuanTokenizer
        if args.load is None:
            tokenizer = BaichuanTokenizer.from_pretrained(
                'baichuan-inc/Baichuan-13B-Base',
                model_max_length=args.seq_length,
                padding_side='right',
                use_fast=False,
            )
        else:
            tokenizer = BaichuanTokenizer.from_pretrained(
                args.load,
                model_max_length=args.seq_length,
                padding_side='right',
                use_fast=False,
            )
        DEFAULT_PAD_TOKEN = '[PAD]'
        DEFAULT_EOS_TOKEN = '</s>'
        DEFAULT_BOS_TOKEN = '<s>'
        DEFAULT_UNK_TOKEN = '<unk>'

        special_tokens_dict = dict()
        if not tokenizer.pad_token:
            special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN
        if not tokenizer.eos_token:
            special_tokens_dict['eos_token'] = DEFAULT_EOS_TOKEN
        if not tokenizer.bos_token:
            special_tokens_dict['bos_token'] = DEFAULT_BOS_TOKEN
        if not tokenizer.unk_token:
            special_tokens_dict['unk_token'] = DEFAULT_UNK_TOKEN
        tokenizer.add_special_tokens(special_tokens_dict)
        args.padded_vocab_size = tokenizer.vocab_size + args.extra_vocab_size

    elif args.patch_tokenizer_type == 'QwenTokenizer':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.load,
            model_max_length=args.seq_length,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens(special_tokens_dict=dict(pad_token="<|extra_0|>"))
        tokenizer.eos_token_id = tokenizer.eod_id
        args.padded_vocab_size = tokenizer.vocab_size + args.extra_vocab_size

    elif args.patch_tokenizer_type == 'QwenVLTokenizer':
        from .tokenization_qwen_vl import QWenTokenizer
        tokenizer = QWenTokenizer.from_pretrained(
            args.load,
            model_max_length=args.seq_length,
            padding_side="right",
            use_fast=False,
            trust_remote_code=False,
        )

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens(special_tokens_dict=dict(pad_token="<|extra_0|>"))
        tokenizer.eos_token_id = tokenizer.eod_id

        args.padded_vocab_size = tokenizer.vocab_size + args.extra_vocab_size

    elif args.patch_tokenizer_type == 'YiTokenizer':
        from .tokenization_yi import YiTokenizer
        if args.load is None:
            tokenizer = YiTokenizer.from_pretrained(
                '01-ai/Yi-6B',
                model_max_length=args.seq_length,
                padding_side='right',
                use_fast=False,
            )
        else:
            tokenizer = YiTokenizer.from_pretrained(
                args.load,
                model_max_length=args.seq_length,
                padding_side='right',
                use_fast=False,
            )
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 2
        args.padded_vocab_size = tokenizer.vocab_size + args.extra_vocab_size

    elif args.patch_tokenizer_type == 'MistralTokenizer':
        print_rank_0('Using Mistral tokenizer.')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.load,
                                                  padding_side='right',
                                                  use_fast=False,)
        tokenizer.pad_token_id = 0
        args.padded_vocab_size = tokenizer.vocab_size + args.extra_vocab_size

    elif args.patch_tokenizer_type == 'BloomTokenizerFromCustom':
        print_rank_0('Using Customized Bloom tokenizer.')
        from transformers import BloomTokenizerFast as BloomTokenizer
        tokenizer = BloomTokenizer.from_pretrained(args.load)
        if 'mg' not in args.load:
            args.padded_vocab_size = 134298
        else:
            args.padded_vocab_size = _vocab_size_with_padding(
                tokenizer.vocab_size, args)
    elif args.patch_tokenizer_type == 'StarcoderTokenizerFromHF':
        print_rank_0('Using Starcoder tokenizer.')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.load)
        tokenizer.pad_token = tokenizer.eos_token
        args.padded_vocab_size = 49152
    elif args.patch_tokenizer_type == 'GPT2BPETokenizer':
        from megatron import get_tokenizer
        tokenizer = get_tokenizer()

    elif args.patch_tokenizer_type == 'VicunaTokenizerFromHF':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.load,
                                                  model_max_length=args.seq_length,
                                                  padding_side="right",
                                                  use_fast=False)
        tokenizer.pad_token = tokenizer.unk_token
        args.padded_vocab_size = 32000

    else:
        raise NotImplementedError('{} tokenizer is not '
                                  'implemented.'.format(
                                      args.patch_tokenizer_type))

    if args.patch_tokenizer_type != 'IcetkGLM130BTokenizer' and\
            args.patch_tokenizer_type != 'GPT2BPETokenizer' and\
            args.patch_tokenizer_type != 'MistralTokenizer':
        tokenizer.eod = tokenizer.eos_token_id

    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = tokenizer
    return _GLOBAL_TOKENIZER


def get_tokenizer():
    """Return tokenizer."""
    return _GLOBAL_TOKENIZER

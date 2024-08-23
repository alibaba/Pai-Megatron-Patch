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

from transformers import AutoTokenizer

def _vocab_size_with_padding(orig_vocab_size, args):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = args.make_vocab_size_divisible_by * \
        args.tensor_model_parallel_size
    while (after % multiple) != 0:
        after += 1
    if args.rank == 0:
        print(' > padded vocab (size: {}) with {} dummy tokens '
              '(new size: {})'.format(
                  orig_vocab_size, after - orig_vocab_size, after), flush=True)
    return after

_GLOBAL_TOKENIZER = None

def get_tokenizer():
    """Return tokenizer."""
    return _GLOBAL_TOKENIZER

def build_tokenizer(args):

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
        tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm-6b',
                                                  trust_remote_code=True)
        args.padded_vocab_size = 130528
    elif args.patch_tokenizer_type == 'GLM10BZHTokenizerFromHF':
        tokenizer = AutoTokenizer.from_pretrained('THUDM/glm-10b-chinese',
                                                  trust_remote_code=True)
        args.padded_vocab_size = 50048
    elif args.patch_tokenizer_type == 'IcetkGLM130BTokenizer':
        from .icetk_glm130b_tokenizer import _IceTokenizer
        tokenizer = _IceTokenizer()
        args.padded_vocab_size = 150528
    elif args.patch_tokenizer_type == 'OPTTokenizer':
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

    elif args.patch_tokenizer_type == 'LLamaTokenizer':
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
        tokenizer = AutoTokenizer.from_pretrained(
            args.load,
            model_max_length=args.seq_length,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens(special_tokens_dict=dict(pad_token="<|extra_0|>"))
        if hasattr(tokenizer, 'eod_id'):
            tokenizer.eos_token_id = tokenizer.eod_id
        args.padded_vocab_size = tokenizer.vocab_size + args.extra_vocab_size

    elif args.patch_tokenizer_type == 'Qwen2Tokenizer':
        from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
        class _Qwen2Tokenizer(MegatronTokenizer):
            def __init__(self, tokenizer_path, extra_vocab_size):
                super().__init__(tokenizer_path)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    padding_side="right",
                    use_fast=False,
                    trust_remote_code=True
                )
                self.extra_vocab_size = extra_vocab_size
                self.tokenizer.add_special_tokens(special_tokens_dict=dict(pad_token="<|extra_0|>"))
                self.tokenizer.add_special_tokens(special_tokens_dict=dict(sep_token="<|extra_1|>"))

            def __call__(self, text, return_tensors=None,
                         padding=None, max_length=None, truncation=None, add_special_tokens=None):

                return self.tokenizer(text, return_tensors=return_tensors, padding=padding,
                        max_length=max_length, truncation=truncation, add_special_tokens=add_special_tokens)

            @property
            def vocab_size(self):
                return len(self.tokenizer.encoder) + self.extra_vocab_size

            @property
            def vocab(self):
                return self.tokenizer.encoder

            @property
            def inv_vocab(self):
                return self.tokenizer.decoder

            def tokenize(self, text):
                return self.tokenizer.encode(text)

            def detokenize(self, token_ids):
                return self.tokenizer.decode(token_ids)

            @property
            def eod(self):
                return self.tokenizer.eos_token_id

            @property
            def eos_token(self):
                return self.tokenizer.eos_token

            @property
            def pad_token_id(self):
                return self.tokenizer.pad_token_id

            @property
            def eos_token_id(self):
                return self.tokenizer.eos_token_id

            @property
            def sep_token_id(self):
                return self.tokenizer.sep_token_id

        tokenizer = _Qwen2Tokenizer(args.load, args.extra_vocab_size)
        args.padded_vocab_size = tokenizer.vocab_size


    elif args.patch_tokenizer_type == 'DeepSeekV2Tokenizer':
        from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
        class _DeepSeekV2Tokenizer(MegatronTokenizer):
            def __init__(self, tokenizer_path, extra_vocab_size):
                super().__init__(tokenizer_path)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    trust_remote_code=True
                )
                self.extra_vocab_size = extra_vocab_size

            def __call__(self, text, return_tensors=None,
                         padding=None, max_length=None, truncation=None, add_special_tokens=None):

                return self.tokenizer(text, return_tensors=return_tensors, padding=padding,
                        max_length=max_length, truncation=truncation, add_special_tokens=add_special_tokens)

            @property
            def vocab_size(self):
                return len(self.tokenizer) + self.extra_vocab_size - 2

            @property
            def vocab(self):
                return self.tokenizer.encoder

            @property
            def inv_vocab(self):
                return self.tokenizer.decoder

            def tokenize(self, text):
                return self.tokenizer.encode(text)

            def detokenize(self, token_ids):
                return self.tokenizer.decode(token_ids)

            @property
            def eod(self):
                return self.tokenizer.eos_token_id

            @property
            def eos_token(self):
                return self.tokenizer.eos_token

            @property
            def pad_token_id(self):
                return self.tokenizer.pad_token_id

            @property
            def eos_token_id(self):
                return self.tokenizer.eos_token_id

        tokenizer = _DeepSeekV2Tokenizer(args.load, args.extra_vocab_size)
        args.padded_vocab_size = tokenizer.vocab_size

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
        tokenizer = AutoTokenizer.from_pretrained(args.load,
                                                  padding_side='right',
                                                  use_fast=False,)
        tokenizer.pad_token_id = 0
        args.padded_vocab_size = tokenizer.vocab_size + args.extra_vocab_size

    elif args.patch_tokenizer_type == 'BloomTokenizerFromCustom':
        from transformers import BloomTokenizerFast as BloomTokenizer
        tokenizer = BloomTokenizer.from_pretrained(args.load)
        if 'mg' not in args.load:
            args.padded_vocab_size = 134298
        else:
            args.padded_vocab_size = _vocab_size_with_padding(
                tokenizer.vocab_size, args)
    elif args.patch_tokenizer_type == 'StarcoderTokenizerFromHF':
        tokenizer = AutoTokenizer.from_pretrained(args.load)
        tokenizer.pad_token = tokenizer.eos_token
        args.padded_vocab_size = 49152

    elif args.patch_tokenizer_type == 'GPT2BPETokenizer':
        from megatron import get_tokenizer
        tokenizer = get_tokenizer()

    elif args.patch_tokenizer_type == 'LLama3Tokenizer':
        from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
        class _LLama3Tokenizer(MegatronTokenizer):
            def __init__(self, tokenizer_path, extra_vocab_size):
                super().__init__(tokenizer_path)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    padding_side="right",
                    use_fast=False,
                    trust_remote_code=True
                )
                self.extra_vocab_size = extra_vocab_size
                # NOTE: Add sep and pad token for LLaMA 3.1
                if self.tokenizer.pad_token is None:
                    self.tokenizer.add_special_tokens(special_tokens_dict=dict(pad_token="<|finetune_right_pad_id|>"))
                    self.tokenizer.add_special_tokens(special_tokens_dict=dict(sep_token="<|reserved_special_token_0|>"))

            def __call__(self, text, return_tensors=None,
                         padding=None, max_length=None, truncation=None, add_special_tokens=None):

                return self.tokenizer(text, return_tensors=return_tensors, padding=padding,
                        max_length=max_length, truncation=truncation, add_special_tokens=add_special_tokens)

            @property
            def vocab_size(self):
                return self.tokenizer.vocab_size + self.extra_vocab_size

            @property
            def vocab(self):
                return self.tokenizer.encoder

            @property
            def inv_vocab(self):
                return self.tokenizer.decoder

            def tokenize(self, text):
                return self.tokenizer.encode(text)

            def detokenize(self, token_ids):
                return self.tokenizer.decode(token_ids)

            @property
            def eod(self):
                return self.tokenizer.eos_token_id

            @property
            def sep_token_id(self):
                return self.tokenizer.sep_token_id

            @property
            def eos_token(self):
                return self.tokenizer.eos_token

            @property
            def pad_token_id(self):
                return self.tokenizer.pad_token_id

            @property
            def eos_token_id(self):
                return self.tokenizer.eos_token_id

        tokenizer = _LLama3Tokenizer(args.load, args.extra_vocab_size)
        args.padded_vocab_size = tokenizer.vocab_size

    elif args.patch_tokenizer_type == 'VicunaTokenizerFromHF':
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


    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = tokenizer
    return _GLOBAL_TOKENIZER


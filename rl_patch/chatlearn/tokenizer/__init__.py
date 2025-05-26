# Copyright 2024 Alibaba Group Holding Limited. All Rights Reserved.
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
# ==============================================================================

import math
from transformers import AutoTokenizer, AutoProcessor
from megatron.training import get_args

def get_tokenizer():
    """Return tokenizer."""
    return _GLOBAL_TOKENIZER
def _vocab_size_with_padding(orig_vocab_size, make_vocab_size_divisible_by, tensor_model_parallel_size):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = make_vocab_size_divisible_by * tensor_model_parallel_size
    after = int(math.ceil(after / multiple) * multiple)
    return after

def build_tokenizer(args):
    patch_tokenizer_type = args.models['policy'].args_dict['patch_tokenizer_type']

    if patch_tokenizer_type == 'DeepSeekV2Tokenizer':
        from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
        class _DeepSeekV2Tokenizer(MegatronTokenizer):
            def __init__(self, tokenizer_path, extra_vocab_size=0):
                super().__init__(tokenizer_path)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    padding_side="right",
                    trust_remote_code=True
                )
                self.extra_vocab_size = extra_vocab_size

                if self.tokenizer.chat_template is None:
                    self.tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ message['content'] + '\n\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
                    try:
                        test_conversation = [
                            {'role': 'user', 'content': 'hello world'}
                        ]
                        self.apply_chat_template(test_conversation)
                    except Exception:
                        # the default chat_template is invalid, assume user will not do SFT
                        self.tokenizer.chat_template = None

            def __call__(self, text, return_tensors=None,
                         padding=None, max_length=None, truncation=None, add_special_tokens=None):

                return self.tokenizer(text, return_tensors=return_tensors, padding=padding,
                        max_length=max_length, truncation=truncation, add_special_tokens=add_special_tokens)

            def apply_chat_template(self, conversations, tokenize:bool=True, **kwargs):
                return self.tokenizer.apply_chat_template(conversations, tokenize=tokenize, **kwargs)

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

            @property
            def eod_id(self):
                return self.tokenizer.pad_token_id


        tokenizer_path = args.models['policy'].args_dict['tokenizer']
        tokenizer = _DeepSeekV2Tokenizer(tokenizer_path, extra_vocab_size=args.models['policy'].args_dict['extra_vocab_size'])

    elif patch_tokenizer_type == 'Qwen2Tokenizer':
        from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
        class _Qwen2Tokenizer(MegatronTokenizer):
            def __init__(self, tokenizer_path, extra_vocab_size=0):
                super().__init__(tokenizer_path)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    padding_side="right",
                    use_fast=False,
                    trust_remote_code=True
                )
                self.extra_vocab_size = extra_vocab_size
                self.tokenizer.add_special_tokens(special_tokens_dict=dict(pad_token="<|extra_0|>"))

                if self.tokenizer.chat_template is None:
                    self.tokenizer.chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
                    try:
                        test_conversation = [
                            {'role': 'user', 'content': 'hello world'}
                        ]
                        self.apply_chat_template(test_conversation)
                    except Exception:
                        # the default chat_template is invalid, assume user will not do SFT
                        self.tokenizer.chat_template = None

            def __call__(self, text, return_tensors=None,
                         padding=None, max_length=None, truncation=None, add_special_tokens=None):

                return self.tokenizer(text, return_tensors=return_tensors, padding=padding,
                                      max_length=max_length, truncation=truncation,
                                      add_special_tokens=add_special_tokens)

            def apply_chat_template(self, conversations):
                return self.tokenizer.apply_chat_template(conversations)

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
            def eod_id(self):
                return self.tokenizer.pad_token_id

        tokenizer_path = args.models['policy'].args_dict['tokenizer']
        tokenizer = _Qwen2Tokenizer(tokenizer_path, extra_vocab_size=args.models['policy'].args_dict['extra_vocab_size'])

    else:
        raise NotImplementedError('{} tokenizer is not '
                                  'implemented.'.format(patch_tokenizer_type))

    if getattr(args, "padded_vocab_size", None) is None:
        make_vocab_size_divisible_by = args.models['policy'].args_dict['make_vocab_size_divisible_by']
        tensor_model_parallel_size = args.models['policy'].args_dict['tensor_model_parallel_size']
        padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size, make_vocab_size_divisible_by,
                                                     tensor_model_parallel_size)
        args.models['policy'].args_dict['padded_vocab_size'] = padded_vocab_size
        get_args().padded_vocab_size = padded_vocab_size

    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = tokenizer
    return _GLOBAL_TOKENIZER


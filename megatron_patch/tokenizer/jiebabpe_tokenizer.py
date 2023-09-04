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

from megatron.tokenizer.tokenizer import AbstractTokenizer
from tokenizers import Tokenizer


class JiebaBPETokenizer(AbstractTokenizer):
    """SentencePiece BPE tokenizer with Jieba integration"""
    def __init__(self, tokenizer_json_file):
        name = 'Jieba BPE Tokenizer'
        super().__init__(name)

        self.tokenizer = Tokenizer.from_file(tokenizer_json_file)
        self.eod_id = self.tokenizer.token_to_id('<|endoftext|>')

        try:
            import rjieba
        except ImportError:
            raise ImportError(
                'You need to install rjieba to use JiebaTokenizer. '
                'See https://pypi.org/project/rjieba/ for installation.')
        self.jieba = rjieba
        self.new_line = self.vocab['\n']
        self.sep_token = self.vocab['<sep>']

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def vocab(self):
        return self.tokenizer.get_vocab(with_added_tokens=True)

    @property
    def inv_vocab(self):
        vocab = self.vocab
        inv_vocab = dict()
        for key, val in vocab.items():
            inv_vocab[val] = key
        return inv_vocab

    def tokenize(self, text, is_code=False, only_cut=False):
        if only_cut:
            seg_list = [x for x in self.jieba.cut(text)]
            return seg_list
        if not is_code:
            seg_list = [x for x in self.jieba.cut(text)]
            return self.tokenizer.encode(seg_list,
                                         is_pretokenized=True,
                                         add_special_tokens=True).ids
        else:
            return self.tokenizer.encode(text,
                                         is_pretokenized=False,
                                         add_special_tokens=True).ids

    def detokenize(self, token_ids):
        text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        return text

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.encode(tokens,
                                     is_pretokenized=True,
                                     add_special_tokens=True).ids

    @property
    def eod(self):
        return self.eod_id

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

import io
import copy
import json
import torch
from megatron import get_args

from megatron_patch.tokenizer import get_tokenizer

class MistralRawDataset(torch.utils.data.Dataset):
    def __init__(self, path, max_padding_length):
        args = get_args()
        self.args = args
        self.tokenizer = get_tokenizer()
        self.IGNORE_INDEX = self.tokenizer.pad_token_id
        if "-Pretrain" in args.dataset:
            self.max_padding_length = max_padding_length + 1
        else:
            self.max_padding_length = max_padding_length
        """
        PROMPT_DICT = {
            'prompt_input':
            ('Below is an instruction that describes a task,'
             ' paired with an input that provides further context. '
             'Write a response that appropriately completes the request.\n\n'
             '### Instruction:\n{instruction}'
             '\n\n### Input:\n{input}\n\n### Response:'),
            'prompt_no_input':
            ('Below is an instruction that describes a task. '
             'Write a response that appropriately completes the request.\n\n'
             '### Instruction:\n{instruction}\n\n### Response:'),
        }
        """
        PROMPT_DICT = {
            'prompt_input': ('[INST]{instruction} {input}[/INST]'),
            'prompt_no_input':('[INST]{instruction}[/INST]'),
        }

        list_data_dict = self.jload(path[0])
        prompt_input, prompt_no_input = PROMPT_DICT[
            'prompt_input'], PROMPT_DICT['prompt_no_input']

        sources = [
            prompt_input.format_map(example) if example.get('input', '') != ''
            else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        if 'output' in list_data_dict[0].keys():
            temp = 'output'
        elif 'content' in list_data_dict[0].keys():
            temp = 'content'

        targets = [
            f"{example[temp]}{self.tokenizer.eos_token}"
            for example in list_data_dict
        ]
        data_dict = self.preprocess(sources, targets, self.tokenizer)

        self.input_ids = data_dict['input_ids']
        self.labels = data_dict['labels']
        self.samples = []
        for inputs, labels in zip(self.input_ids, self.labels):
            self.samples.append([inputs, labels])

        print('  >> total number of samples: {}'.format(len(self.samples)))

    def _make_r_io_base(self, f, mode: str):
        if not isinstance(f, io.IOBase):
            f = open(f, mode=mode, encoding='utf-8')
        return f

    def jload(self, f, mode='r'):
        """
        Load a .json file into a dictionary.
        Args:
            f: The file object or string representing the file path.
            mode: The mode in which to open the file (e.g., 'r', 'w', 'a').
        Returns:
            A dictionary containing the contents of the JSON file.
        """
        f = self._make_r_io_base(f, mode)
        jdict = json.load(f)
        f.close()
        return jdict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]
        return self.gpt_convert_example_to_feature(raw_sample)

    def preprocess(self, sources, targets, tokenizer):
        """
        Preprocess the data by tokenizing.
        Args:
            sources (List[str]): a list of source strings
            targets (List[str]): a list of target strings
            tokenizer (Tokenizer): a tokenizer object used for tokenization
        Returns:
            dict: a dictionary containing the input_ids and labels for the examples
        """

        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [
            self.tokenize(strings, tokenizer)
            for strings in (examples, sources)
        ]
        input_ids = examples_tokenized['input_ids']
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels,
                                     sources_tokenized['input_ids_lens']):
            label[:source_len] = self.IGNORE_INDEX
        return dict(input_ids=input_ids, labels=labels)

    def tokenize(self, strings, tokenizer):
        """
        Tokenize a list of strings.
        Args:
            strings (List[str]): a list of strings to be tokenized
            tokenizer (Tokenizer): a tokenizer object used for tokenization
        Returns:
            dict: a dictionary containing the input_ids and labels for the tokenized strings
        """
        tokenized_list = [
            tokenizer.encode(text) for text in strings
        ]
        new_tokenzied_list = []
        for tokenized in tokenized_list:
            if self.max_padding_length <= len(tokenized):
                new_tokenzied_list.append(torch.tensor(tokenized[:self.max_padding_length]))
            else:
                padding_len = self.max_padding_length - len(tokenized)
                new_tokenized = torch.cat(
                    [torch.tensor(tokenized), torch.full((padding_len,), tokenizer.pad_token_id, dtype=torch.int)])
                new_tokenzied_list.append(new_tokenized)

        input_ids = labels = [
            tokenized for tokenized in new_tokenzied_list
        ]
        input_ids_lens = labels_lens = [
            (tokenized != tokenizer.pad_token_id).sum().item()
            for tokenized in new_tokenzied_list
        ]

        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def gpt_convert_example_to_feature(self, sample):
        """
        Convert a single sample containing input_id, label and loss_mask into a format suitable for GPT training.
        """
        input_ids, labels = sample
        train_sample = {
            'input_ids': input_ids,
            'labels': labels
        }

        return train_sample
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

import copy
import io
import json
import numpy as np
import torch

from megatron_patch.tokenizer import get_tokenizer

class StarcoderRawDataset(torch.utils.data.Dataset):
    """Starcoder dataset class"""
    def __init__(self, datapath, max_padding_length):
        """
        Args:
            datapath (str): The path of the dataset.
            max_padding_length (int): The maximum length to pad the input sequences to.
        """

        self.IGNORE_INDEX = -100
        self.tokenizer = get_tokenizer()
        self.max_padding_length = max_padding_length
        PROMPT_DICT = {
            'prompt_input':
            ('<|user|>{instruction}{input}\n\n<|bot|>'),
            'prompt_no_input':
            ('<|user|>{instruction}\n\n<|bot|>'),
        }

        list_data_dict = self.jload(datapath)
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
        """Load a .json file into a dictionary."""
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
        """Preprocess the data by tokenizing."""
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
        This function tokenizes the source strings given the tokenizer and returns a dictionary containing the
        tokenized inputs and labels.
        Args:
            strings (List[str]): The list of input strings.
            tokenizer (PreTrainedTokenizer): The tokenizer to use.
            max_input_length (Optional[int]): The maximum length of the input sequences.
            max_target_length (Optional[int]): The maximum length of the target sequences.
        Returns:
            Dict[str, Any]: A dictionary containing input_ids, labels, input_ids_lens and labels_lens.
        """

        tokenized_list = [
            tokenizer(
                text,
                return_tensors='np',
                padding='max_length',
                max_length=self.max_padding_length,
                truncation=True,
            ) for text in strings
        ]
        input_ids = labels = [
            tokenized.input_ids[0] for tokenized in tokenized_list
        ]
        input_ids_lens = labels_lens = [
            (tokenized.input_ids != tokenizer.pad_token_id).sum().item()
            for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def gpt_convert_example_to_feature(self, sample):
        input_ids, labels = sample
        loss_mask = np.ones(labels.shape, dtype=np.int64)
        loss_mask[labels == self.IGNORE_INDEX] = 0
        loss_mask[labels == self.tokenizer.pad_token_id] = 0
        train_sample = {
            'input_ids': input_ids,
            'labels': labels,
            'loss_mask': loss_mask
        }

        return train_sample

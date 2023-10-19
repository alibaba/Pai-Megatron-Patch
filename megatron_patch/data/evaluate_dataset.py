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
import math
import os
from bisect import bisect_right
from itertools import accumulate
import numpy as np
import torch

from megatron import get_args
from megatron_patch.tokenizer import get_tokenizer

class GLM130BDataset(torch.utils.data.Dataset):
    """A class for processing a GLM130B text dataset"""
    def __init__(self, path, tokenizer, max_seq_length, generation_length):
        """
        Initializes the dataset.
        Args:
            path(str): The path of the dataset file.
            tokenizer(object): The tokenizer object.
            max_seq_length(int): The maximum length of sequences.
            generation_length(int): The length of generated sequence.
        """
        self.path = path
        self.max_seq_length = max_seq_length
        self.generation_length = generation_length
        self.dtype = np.int64
        self.tokenizer = tokenizer

        self.tokenizer = get_tokenizer()
        self.mask_id = self.tokenizer.get_command('[MASK]')
        self.gmask_id = self.tokenizer.get_command('[gMASK]')
        self.data = []
        self.process_single_file(self.path)

    def process_single_file(self, path):
        """
        Processes a single dataset file.
        Args:
            path(str): The path of the dataset file.
        """
        num_sequences = []
        with open(os.path.join(path), 'r', encoding='utf-8') as file:
            raw_text = file.read()
            tokens = self.tokenizer.tokenize(raw_text)
            self.num_tokenized_tokens = len(tokens)
            self.num_original_tokens = len(raw_text.strip().split(' '))
            self.data.append({
                'raw_text':
                tokens,
                'num_original_tokens':
                len(raw_text.strip().split(' ')),
                'num_sequences':
                max(
                    math.ceil(
                        max(len(tokens) - (self.max_seq_length - 1), 0) /
                        self.generation_length) + 1,
                    1,
                ),
            })
            num_sequences.append(self.data[-1]['num_sequences'])
        self.weights = list(accumulate(num_sequences))
        self.left_weights = [0] + self.weights[:-1]

    def __len__(self):
        """
        Returns the number of sequences in the dataset.
        """
        return self.data[0]['num_sequences']

    def __getitem__(self, idx):
        """
        Returns the item at the given index.
        Args:
            idx (int): The index of the item to return.
        Returns:
            A dictionary containing the following tokens, targets, position_ids, attention_mask and loss_mask.
        """

        document_idx = bisect_right(self.weights, idx)
        idx = idx - self.left_weights[document_idx]
        start_idx = idx * self.generation_length
        end_idx = start_idx + self.max_seq_length - 1  # for additional [gMASK]
        tokens = self.data[document_idx]['raw_text'][start_idx:end_idx]

        mask_id = self.gmask_id
        sop_id = self.tokenizer.get_command('sop')

        if idx == 0:
            prompt, text = [], tokens
        else:
            prompt_length = self.max_seq_length - 1 - self.generation_length
            prompt, text = tokens[:prompt_length], tokens[prompt_length:]

        seq_length = len(prompt) + len(text) + 1
        attention_mask = np.tril(
            np.ones((seq_length, seq_length), dtype=np.int64))
        attention_mask[:len(prompt) + 1, :len(prompt) + 1] = 1
        return {
            'tokens':
            np.array(prompt + [mask_id, sop_id] + text[:-1], dtype=np.int64),
            'targets':
            np.array(prompt + [mask_id] + text, dtype=np.int64),
            'position_ids':
            np.arange(0, seq_length, dtype=np.int64),
            'attention_mask':
            attention_mask < 0.5,
            'loss_mask':
            np.array([0] * (len(prompt) + 1) + [1] * len(text),
                     dtype=np.int64),
        }


class BloomDataset(torch.utils.data.Dataset):
    """A class for processing a Bloom text dataset"""
    def __init__(self, datapaths, tokenizer, max_seq_length):
        """
        Initializes the dataset.
        Args:
            path(str): The path of the dataset file.
            tokenizer(object): The tokenizer object.
            max_seq_length(int): The maximum length of sequences.
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.prompt = ''
        self.samples = []
        for datapath in datapaths:
            self.samples.extend(
                self.process_samples_from_single_path(datapath))
        print('  >> total number of samples: {}'.format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        idx = 0
        raw_sample = self.samples[idx]
        return self.gpt_convert_example_to_feature(raw_sample, self.tokenizer,
                                                   self.max_seq_length)

    def truncate(self, tokenizer, array, max_length):
        """
        Truncates an array to a maximum length or pads it with zeros if its length is less than `max_length`.
        Args:
            tokenizer: The tokenizer used to encode the input.
            array: The numpy array to truncate or pad.
            max_length: The maximum length of the array.
        Returns:
            A numpy array of length `max_length` containing the contents of `array`, truncated if necessary or padded with zeros.
        """

        if len(array) < max_length:
            return np.pad(array, (0, max_length - len(array)),
                          constant_values=tokenizer.eod)
        else:
            return array[:max_length]

    def process_samples_from_single_path(self, filename):
        """
        Process a single file containing prompt-answer pairs and return a list of samples.
        """

        print(' > Processing {} ...'.format(filename))
        samples = []
        total = 0
        with open(filename, encoding='utf-8-sig') as f:
            for example in f:
                text = json.loads(example)['text']
                sample = {
                    'prompt':
                    text + '</s>' if not text.endswith('</s>') else text,
                    'answer': text,
                }
                total += 1
                samples.append(sample)

        print(' >> processed {} samples.'.format(len(samples)))
        return samples

    def gpt_convert_example_to_feature(self, sample, tokenizer,
                                       max_seq_length):
        """
        Convert a single sample containing a prompt-answer pair into a format suitable for GPT training.
        """

        tokens = tokenizer(sample['prompt'])
        input_ids = tokens['input_ids']
        input_ids = self.truncate(tokenizer, input_ids, max_seq_length + 1)
        train_sample = {'input_ids': np.array(input_ids)}
        return train_sample


class LLamaDataset(torch.utils.data.Dataset):
    """A class for processing a LLama text dataset"""
    def __init__(self, path, tokenizer, max_padding_length):
        """
        Initializes the dataset.
        Args:
            path(str): The path of the dataset file.
            tokenizer(object): The tokenizer object.
            max_padding_length(int): The maximum length of the padding to the sequences.
        """
        self.IGNORE_INDEX = -100
        self.tokenizer = tokenizer
        self.max_padding_length = max_padding_length
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

        list_data_dict = self.jload(path)
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
            f"{example[temp]}{tokenizer.eos_token}"
            for example in list_data_dict
        ]

        data_dict = self.preprocess(sources, targets, tokenizer)

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
        idx = 0
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
        """
        Convert a single sample containing input_id, label and loss_mask into a format suitable for GPT training.
        """
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


def build_evaluation_dataset(dataset):
    """
    Helper function to select and build the appropriate dataset object for evaluation.
    Args:
        dataset (str): The name of the dataset to use.
    Returns:
        Dataset: The evaluation dataset object.
    """
    args = get_args()
    tokenizer = get_tokenizer()

    if dataset == 'GLM130B-WIKITEXT103':
        val_dataset = GLM130BDataset(args.data_path[0], tokenizer,
                                     args.seq_length, args.generation_length)
        return val_dataset

    elif dataset == 'LLama-SFT':
        val_dataset = LLamaDataset(args.data_path[0], tokenizer,
                                   args.max_padding_length)
        return val_dataset

    elif dataset == 'Bloom-SFT':
        val_dataset = BloomDataset(args.data_path, tokenizer,
                                   args.max_padding_length)

        return val_dataset

    raise NotImplementedError('dataset {} is not implemented.'.format(dataset))

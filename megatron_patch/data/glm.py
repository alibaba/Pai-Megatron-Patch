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

import random
import json
import math
import os
from bisect import bisect_right
from itertools import accumulate
import numpy as np
import torch

from megatron_patch.tokenizer import get_tokenizer


class GLM130BRawDataset(torch.utils.data.Dataset):
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

class GLMRawDataset(torch.utils.data.Dataset):
    """GLM dataset class."""
    def __init__(self, datapaths, tokenizer, max_source_seq_length,
                 max_target_seq_length):
        self.tokenizer = tokenizer
        self.prompt = ''
        self.samples = []
        self.random = random.Random(1234)
        self.blank_maskratio = 0.1
        self.max_src_length, self.max_tgt_length =\
            max_source_seq_length, max_target_seq_length
        for datapath in datapaths:
            self.samples.extend(
                self.process_samples_from_single_path(datapath))
        print('  >> total number of samples: {}'.format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]
        return self.gpt_convert_example_to_feature(raw_sample, self.tokenizer)

    def process_samples_from_single_path(self, filename):
        print(' > Processing {} ...'.format(filename))
        samples = []
        total = 0
        with open(filename, encoding='utf-8-sig') as f:
            for example in f:
                json_obj = json.loads(example)
                prompt = json_obj['question']
                answer = json_obj['answer']
                source_tokenized_text = self.tokenizer._tokenize(prompt)
                target_tokenized_text = self.tokenizer._tokenize(answer)
                sample = {
                    'source': ' '.join(source_tokenized_text),
                    'target': ' '.join(target_tokenized_text),
                }
                total += 1
                samples.append(sample)

        print(' >> processed {} samples.'.format(len(samples)))
        random.shuffle(samples)
        return samples

    def mask_text(self, text):
        tokens = text.split()
        mask_ratio = self.blank_maskratio
        n = len(tokens)
        indices = sorted(self.random.sample(range(n), int(n * mask_ratio)))
        masked_src, masked_tgt = '', []
        for i, idx in enumerate(indices):
            if i == 0 or idx != indices[i - 1] + 1:
                masked_tgt.append('')
            masked_tgt[-1] += ' ' + tokens[idx]
            tokens[idx] = '[MASK]'
        for i, token in enumerate(tokens):
            if i != 0 and token == '[MASK]' and tokens[i - 1] == '[MASK]':
                continue
            masked_src += ' ' + token
        return masked_src, masked_tgt

    def gpt_convert_example_to_feature(self, sample, tokenizer):
        # GLM BlankLMDataset
        source_text = sample['target']
        mask_id = tokenizer.mask_token_id
        sop_id = tokenizer.cls_token_id
        eop_id = tokenizer.eop_token_id
        pad_id = tokenizer.pad_token_id
        masked_src, masked_tgt = self.mask_text(source_text)
        source_text = masked_src

        def pad_to(text, max_len, pad_id):
            if len(text) > max_len:
                text = text[:max_len]
            else:
                text = text + [pad_id] * (max_len - len(text))
            return text

        source_tokens = tokenizer.convert_tokens_to_ids(source_text.split())
        source_tokens = pad_to(source_tokens, self.max_src_length, pad_id)
        sep = len(source_tokens)
        position_ids = list(range(len(source_tokens)))
        block_position_ids = [0] * len(source_tokens)
        mask_positions = [
            i for i, x in enumerate(source_tokens) if x == mask_id
        ]
        assert len(mask_positions) <= len(masked_tgt)
        tokens = source_tokens
        target_ids = [0] * len(source_tokens)
        loss_mask = [0] * len(source_tokens)
        for i, mask_pos in enumerate(mask_positions):
            tgt_text = masked_tgt[i]
            tgt_tokens = tokenizer.convert_tokens_to_ids(tgt_text.split())
            tokens += [sop_id] + tgt_tokens
            target_ids += tgt_tokens + [eop_id]
            loss_mask += [1] * (len(tgt_tokens) + 1)
            position_ids += [mask_pos] * (len(tgt_tokens) + 1)
            block_position_ids += [i + 1 for i in range(len(tgt_tokens) + 1)]

        max_length = self.max_src_length + int(
            self.max_src_length * self.blank_maskratio)

        tokens = pad_to(tokens, max_length, pad_id)
        target_ids = pad_to(target_ids, max_length, pad_id)
        loss_mask = pad_to(loss_mask, max_length, 0)
        position_ids = pad_to(position_ids, max_length, 0)
        block_position_ids = pad_to(block_position_ids, max_length, 0)
        position_ids = [position_ids, block_position_ids]
        train_sample = {
            'text': np.array(tokens, dtype=np.int64),
            'target': np.array(target_ids, dtype=np.int64),
            'attention_mask': np.array(sep, dtype=np.int64),
            'loss_mask': np.array(loss_mask, dtype=np.int64),
            'position_id': np.array(position_ids, dtype=np.int64)
        }

        return train_sample

class ChatGLMRawDataset(torch.utils.data.Dataset):
    """ChatGLM dataset class."""
    def __init__(self, datapaths, max_source_length,
                 max_target_length):
        self.tokenizer = get_tokenizer()
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.samples = []
        for datapath in datapaths:
            self.samples.extend(
                self.process_samples_from_single_path(datapath))
        print('  >> total number of samples: {}'.format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]
        return self.gpt_convert_example_to_feature(raw_sample, self.tokenizer)

    def process_samples_from_single_path(self, filename):
        print(' > Processing {} ...'.format(filename))
        samples = []
        total = 0
        with open(filename, encoding='utf-8-sig') as f:
            for example in f:
                json_obj = json.loads(example)
                content = json_obj['content']
                summary = json_obj['summary']
                sample = {
                    'source': content,
                    'target': summary,
                }
                total += 1
                samples.append(sample)

        print(' >> processed {} samples.'.format(len(samples)))
        random.shuffle(samples)
        return samples

    def gpt_convert_example_to_feature(self, sample, tokenizer):

        prompt, answer = sample['source'], sample['target']
        a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
        b_ids = tokenizer.encode(text=answer, add_special_tokens=False)
        max_seq_length = self.max_source_length + self.max_target_length
        if len(a_ids) > self.max_source_length - 1:
            a_ids = a_ids[:self.max_source_length - 1]

        if len(b_ids) > self.max_target_length - 2:
            b_ids = b_ids[:self.max_target_length - 2]

        input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)
        context_length = input_ids.index(tokenizer.bos_token_id)
        mask_position = context_length - 1
        labels = [-100] * context_length + input_ids[mask_position + 1:]

        pad_len = max_seq_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        labels = labels + [tokenizer.pad_token_id] * pad_len
        labels = [(label if label != tokenizer.pad_token_id else -100)
                  for label in labels]

        train_sample = {
            'input_ids': np.array(input_ids, dtype=np.int64),
            'labels': np.array(labels, dtype=np.int64)
        }

        return train_sample
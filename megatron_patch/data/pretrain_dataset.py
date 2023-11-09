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

from megatron.data.dataset_utils import get_datasets_weights_and_num_samples
from megatron.data.blendable_dataset import BlendableDataset
from megatron import get_args
from megatron import print_rank_0
from megatron.data.gpt_dataset import _build_index_mappings
from megatron.data.gpt_dataset import get_indexed_dataset_
from megatron.data.gpt_dataset import get_train_valid_test_split_

from megatron_patch.tokenizer import get_tokenizer

class GLM130BDataset(torch.utils.data.Dataset):
    """GLM130B dataset class."""
    def __init__(self, path, max_seq_length, generation_length):
        """
        Initializes the dataset with the path to the text file, the maximum length of input sequences,
        and the length of generated text after the prompt.
        """
        self.path = path
        self.max_seq_length = max_seq_length
        self.generation_length = generation_length
        self.dtype = np.int64
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
        return self.data[0]['num_sequences']

    def __getitem__(self, idx):
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
            attention_mask,
            'loss_mask':
            np.array([0] * (len(prompt) + 1) + [1] * len(text),
                     dtype=np.int64),
        }


class GLM130BIdxMapDataset(torch.utils.data.Dataset):
    """GLM130B dataset class for mmap format data"""
    def __init__(self,
                 name,
                 data_prefix,
                 documents,
                 indexed_dataset,
                 num_samples,
                 seq_length,
                 generation_length,
                 seed,
                 return_doc_ids=False):
        """
        Initializes the GLM130BIdxMapDataset class.
        Args:
            name (str): Dataset name.
            data_prefix (str): Path prefix.
            documents (list of int): List of document indices.
            indexed_dataset (object): Indexed dataset object.
            num_samples (int): Number of samples.
            seq_length (int): Maximum sequence length.
            generation_length (int): Generation length (length of generated text).
            seed (int): Random seed.
            return_doc_ids (bool, optional): Whether to return document ids. Defaults to False.
        """

        self.max_seq_length = seq_length
        self.generation_length = generation_length
        self.tokenizer = get_tokenizer()
        self.mask_id = self.tokenizer.get_command('[MASK]')
        self.gmask_id = self.tokenizer.get_command('[gMASK]')

        self.name = name
        self.indexed_dataset = indexed_dataset
        self.return_doc_ids = return_doc_ids

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        # Build index mappings.
        self.doc_idx, self.sample_idx, self.shuffle_idx, self.index_prefix = \
            _build_index_mappings(self.name, data_prefix,
                                  documents, self.indexed_dataset.sizes,
                                  num_samples, seq_length, seed)

    def __len__(self):
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self.sample_idx.shape[0] - 1

    def __getitem__(self, idx):
        # Get the shuffled index.
        idx = self.shuffle_idx[idx]
        # Start and end documents and offsets.
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        # If we are within the same document, just extract the chunk.
        doc_ids = []
        if doc_index_f == doc_index_l:
            doc_ids.append(self.doc_idx[doc_index_f])
            sample = self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                              offset=offset_f,
                                              length=offset_l - offset_f + 1)
        else:
            # Otherwise, get the rest of the initial document.
            doc_ids.append(self.doc_idx[doc_index_f])
            sample_list = [
                self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                         offset=offset_f)
            ]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                doc_ids.append(self.doc_idx[i])
                sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
            # And finally add the relevant portion of last document.
            doc_ids.append(self.doc_idx[doc_index_l])
            sample_list.append(
                self.indexed_dataset.get(self.doc_idx[doc_index_l],
                                         length=offset_l + 1))
            sample = np.concatenate(sample_list)

        tokens = sample[:-2].tolist()
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
            attention_mask,
            'loss_mask':
            np.array([0] * (len(prompt) + 1) + [1] * len(text),
                     dtype=np.int64),
        }


class LLamaDataset(torch.utils.data.Dataset):
    """LLAMA dataset class"""
    def __init__(self, datapath, max_padding_length):
        """
        Args:
            datapath (str): The path of the dataset.
            max_padding_length (int): The maximum length to pad the input sequences to.
        """
        self.tokenizer = get_tokenizer()
        self.IGNORE_INDEX = self.tokenizer.pad_token_id
        self.max_padding_length = max_padding_length
        PROMPT_DICT = {
            'prompt_input':
            ('instruction:{instruction}input:{input}\n'),
            'prompt_no_input':
            ('instruction:{instruction}\n'),
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
                max_length=self.max_padding_length+1,
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

class LLamaIdxMapDataset(torch.utils.data.Dataset):
    """LLAMA dataset class for mmap format data"""
    def __init__(self,
                 name,
                 data_prefix,
                 documents,
                 indexed_dataset,
                 num_samples,
                 seed,
                 max_padding_length,
                 return_doc_ids=False):

        # self.IGNORE_INDEX = -100
        args = get_args()
        self.tokenizer = get_tokenizer()
        self.max_padding_length = max_padding_length

        self.name = name
        self.indexed_dataset = indexed_dataset
        self.return_doc_ids = return_doc_ids
        self.split = args.split
        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        # Build index mappings.
        try:
            self.doc_idx, self.sample_idx, self.shuffle_idx, self.index_prefix = \
                _build_index_mappings(self.name, data_prefix,
                                  documents, self.indexed_dataset.sizes,
                                  num_samples, self.max_padding_length, seed)
        except:
            self.doc_idx, self.sample_idx, self.shuffle_idx, self.desc, self.desc_hash = \
                _build_index_mappings(self.name, data_prefix,
                                  documents, self.indexed_dataset.sizes,
                                  self.split, num_samples, self.max_padding_length, seed,
                                  data_cache_path=None)

    def __len__(self):
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self.sample_idx.shape[0] - 1

    def __getitem__(self, idx):
        # Get the shuffled index.
        idx = self.shuffle_idx[idx]
        # Start and end documents and offsets.
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        # If we are within the same document, just extract the chunk.
        doc_ids = []

        if doc_index_f == doc_index_l:
            doc_ids.append(self.doc_idx[doc_index_f])
            sample = self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                              offset=offset_f,
                                              length=offset_l - offset_f + 1)
        else:
            # Otherwise, get the rest of the initial document.
            doc_ids.append(self.doc_idx[doc_index_f])
            sample_list = [
                self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                         offset=offset_f)
            ]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                doc_ids.append(self.doc_idx[i])
                sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
            # And finally add the relevant portion of last document.
            doc_ids.append(self.doc_idx[doc_index_l])
            sample_list.append(
                self.indexed_dataset.get(self.doc_idx[doc_index_l],
                                         length=offset_l + 1))
            sample = np.concatenate(sample_list)

        tokens = sample.tolist()
        sample = []
        sample.append(np.array(tokens))
        sample.append(np.array(tokens))

        return self.gpt_convert_example_to_feature(sample)

    def gpt_convert_example_to_feature(self, sample):
        input_ids, labels = sample
        loss_mask = np.ones(labels.shape, dtype=np.int64)
        loss_mask[labels == self.tokenizer.bos_token_id] = 0
        loss_mask[labels == self.tokenizer.pad_token_id] = 0
        train_sample = {
            'input_ids': input_ids,
            'labels': labels,
            'loss_mask': loss_mask
        }

        return train_sample


class StarcoderDataset(torch.utils.data.Dataset):
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


def build_pretrain_glm130b_datasets_from_idxmap(data_prefix,
                                                data_impl,
                                                splits_string,
                                                train_valid_test_num_samples,
                                                seq_length,
                                                generation_length,
                                                seed,
                                                skip_warmup,
                                                return_doc_ids=False):
    """
    Build train, valid, and test datasets for pretraining a GLM130B model on mmap format data.
    Args:
        data_prefix (str): common prefix added to the front of files.
        data_impl (str): implementation of the data loader.
        splits_string (str): string specifying the dataset splits.
        train_valid_test_num_samples (List[int]): Number of training, validation, and test samples.
        seq_length (int): the sequence length of the input sequence.
        generation_length (int): the length of generated sequences.
        seed (int): seed for the random number generator.
        skip_warmup (bool): whether to skip the warmup period.
        return_doc_ids (bool): whether to return document IDs along with the input and target sequences.
    Returns:
        Tuple[Optional[GLM130BIdxMapDataset], Optional[GLM130BIdxMapDataset], Optional[GLM130BIdxMapDataset]]:
        The train, validation, and test datasets respectively wrapped in an optional class.
    """
    data_prefix = data_prefix[0]
    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix, skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    def print_split_stats(name, index):
        print_rank_0('    {}:'.format(name))
        print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[index], splits[index + 1],
                                        splits[index + 1] - splits[index]))

    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index],
                                  stop=splits[index + 1],
                                  step=1,
                                  dtype=np.int32)
            dataset = GLM130BIdxMapDataset(
                name, data_prefix, documents, indexed_dataset,
                train_valid_test_num_samples[index], seq_length,
                generation_length, seed, return_doc_ids)
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return (train_dataset, valid_dataset, test_dataset)


def build_pretrain_glm130b_datasets_from_original(data_prefix, max_seq_length,
                                                  generation_length):
    """
    Build train, valid, and test datasets for pretraining a GLM130B model on original format data.
    """
    def build_dataset():

        dataset = GLM130BDataset(data_prefix[0], max_seq_length,
                                 generation_length)

        return dataset

    train_dataset = build_dataset()
    valid_dataset = build_dataset()
    test_dataset = build_dataset()

    return (train_dataset, valid_dataset, test_dataset)


def build_pretrain_llama_datasets_from_idxmap(data_prefix,
                                              max_padding_length,
                                              data_impl,
                                              splits_string,
                                              train_valid_test_num_samples,
                                              seed,
                                              skip_warmup,
                                              return_doc_ids=False):
    """
    Build train, valid, and test datasets for pretraining a LLAMA model on mmap format data.
    Args:
        data_prefix (str): common prefix added to the front of files.
        max_padding_length (int): Maximum sequence length after padding.
        data_impl (str): implementation of the data loader.
        splits_string (str): string specifying the dataset splits.
        train_valid_test_num_samples (Tuple[int, int, int]): Number of training, validation, and test samples.
        seed (int): seed for the random number generator.
        skip_warmup (bool): whether to skip the warmup period.
        return_doc_ids (bool): whether to return document IDs along with the input and target sequences.
    Returns:
        A tuple of three LLamaIdxMapDataset objects: train_dataset, valid_dataset, and test_dataset.
    """
    if len(data_prefix) == 1:
        return _build_train_valid_test_datasets(data_prefix[0],max_padding_length,
                                                data_impl, splits_string,
                                                train_valid_test_num_samples,
                                                seed, skip_warmup, return_doc_ids)

    # Blending dataset.
    # Parse the values.
    output = get_datasets_weights_and_num_samples(data_prefix,
                                                  train_valid_test_num_samples)
    prefixes, weights, datasets_train_valid_test_num_samples = output
    train_num_samples, valid_num_samples, test_num_samples = map(
        sum,
        zip(*datasets_train_valid_test_num_samples)
    )

    # Build individual datasets.
    train_datasets = []
    valid_datasets = []
    test_datasets = []
    for i in range(len(prefixes)):
        train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
            prefixes[i], max_padding_length, data_impl, splits_string,
            datasets_train_valid_test_num_samples[i],
            seed, skip_warmup,
            return_doc_ids)
        if train_ds:
            train_datasets.append(train_ds)
        if valid_ds:
            valid_datasets.append(valid_ds)
        if test_ds:
            test_datasets.append(test_ds)

    # Blend.
    blending_train_dataset = None
    if train_datasets:
        blending_train_dataset = BlendableDataset(train_datasets, weights, train_num_samples)
    blending_valid_dataset = None
    if valid_datasets:
        blending_valid_dataset = BlendableDataset(valid_datasets, weights, valid_num_samples)
    blending_test_dataset = None
    if test_datasets:
        blending_test_dataset = BlendableDataset(test_datasets, weights, test_num_samples)

    return (blending_train_dataset, blending_valid_dataset,
            blending_test_dataset)



def _build_train_valid_test_datasets(data_prefix, max_padding_length, data_impl, splits_string,
                                     train_valid_test_num_samples,
                                     seed, skip_warmup,
                                     return_doc_ids=False):
    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix, skip_warmup)
    total_num_of_documents = indexed_dataset.sizes.shape[0]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)
    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    def print_split_stats(name, index):
        print_rank_0('    {}:'.format(name))
        print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[index], splits[index + 1],
                                        splits[index + 1] - splits[index]))

    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index],
                                  stop=splits[index + 1],
                                  step=1,
                                  dtype=np.int32)
            dataset = LLamaIdxMapDataset(
                name, data_prefix, documents, indexed_dataset,
                train_valid_test_num_samples[index],
                seed, max_padding_length, return_doc_ids)
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return (train_dataset, valid_dataset, test_dataset)


def build_pretrain_llama_datasets_from_original(data_prefix,
                                                max_padding_length):
    """
    Build train, valid, and test datasets for pretraining a LLAMA model on original format data.
    """
    def build_dataset():

        dataset = LLamaDataset(data_prefix[0], max_padding_length)

        return dataset

    train_dataset = build_dataset()
    valid_dataset = build_dataset()
    test_dataset = build_dataset()

    return (train_dataset, valid_dataset, test_dataset)

def build_pretrain_mistral_datasets_from_original(data_prefix,
                                                max_padding_length):
    """
    Build train, valid, and test datasets for pretraining a LLAMA model on original format data.
    """
    def build_dataset():

        dataset = LLamaDataset(data_prefix[0], max_padding_length)

        return dataset

    train_dataset = build_dataset()
    valid_dataset = build_dataset()
    test_dataset = build_dataset()

    return (train_dataset, valid_dataset, test_dataset)


def build_pretrain_falcon_datasets_from_idxmap(data_prefix,
                                               max_padding_length,
                                               data_impl,
                                               splits_string,
                                               train_valid_test_num_samples,
                                               seed,
                                               skip_warmup,
                                               return_doc_ids=False):
    """
    Build train, valid, and test datasets for pretraining a falcon model on mmap format data.
    Args:
        data_prefix (str): common prefix added to the front of files.
        max_padding_length (int): Maximum sequence length after padding.
        data_impl (str): implementation of the data loader.
        splits_string (str): string specifying the dataset splits.
        train_valid_test_num_samples (Tuple[int, int, int]): Number of training, validation, and test samples.
        seed (int): seed for the random number generator.
        skip_warmup (bool): whether to skip the warmup period.
        return_doc_ids (bool): whether to return document IDs along with the input and target sequences.
    Returns:
        A tuple of three LLamaIdxMapDataset objects: train_dataset, valid_dataset, and test_dataset.
    """
    data_prefix = data_prefix[0]
    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix, skip_warmup)
    total_num_of_documents = indexed_dataset.sizes.shape[0]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)
    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    def print_split_stats(name, index):
        print_rank_0('    {}:'.format(name))
        print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[index], splits[index + 1],
                                        splits[index + 1] - splits[index]))

    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index],
                                  stop=splits[index + 1],
                                  step=1,
                                  dtype=np.int32)
            dataset = LLamaIdxMapDataset(
                name, data_prefix, documents, indexed_dataset,
                train_valid_test_num_samples[index],
                seed, max_padding_length, return_doc_ids)
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')


    return (train_dataset, valid_dataset, test_dataset)

def build_pretrain_falcon_datasets_from_original(data_prefix,
                                                 max_padding_length):
    """
    Build train, valid, and test datasets for pretraining a falcon model on original format data.
    """
    def build_dataset():

        dataset = LLamaDataset(data_prefix[0], max_padding_length)

        return dataset

    train_dataset = build_dataset()
    valid_dataset = build_dataset()
    test_dataset = build_dataset()

    return (train_dataset, valid_dataset, test_dataset)


def build_pretrain_starcoder_datasets_from_original(data_prefix,
                                                max_padding_length):
    """
    Build train, valid, and test datasets for pretraining a LLAMA model on original format data.
    """
    def build_dataset():

        dataset = StarcoderDataset(data_prefix[0], max_padding_length)

        return dataset

    train_dataset = build_dataset()
    valid_dataset = build_dataset()
    test_dataset = build_dataset()

    return (train_dataset, valid_dataset, test_dataset)

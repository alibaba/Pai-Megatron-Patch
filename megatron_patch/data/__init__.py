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

import numpy as np
try:
    from megatron import get_args
except:
    from megatron.training import get_args

from megatron_patch.tokenizer import build_tokenizer
from .llama import LLamaRawDataset
from .bloom import BloomRawDataset
from .llava.mm_pretrain_dataset import LazySupervisedDataset as LLavaSupervisedDataset
from .qwen_vl import LazySupervisedDataset as QwenVLSupervisedDataset
from .glm import ChatGLMRawDataset
from .starcoder import StarcoderRawDataset

def build_evaluation_dataset(dataset):

    args = get_args()
    build_tokenizer(args)
    if dataset == 'LLama-SFT' or dataset == 'LLama-Pretrain-Raw':
        val_dataset = LLamaRawDataset(args.valid_data_path, args.max_padding_length)
        return val_dataset
    else:
        raise NotImplementedError('dataset {} is not implemented.'.format(dataset))

def build_finetune_dataset(dataset):

    args = get_args()
    build_tokenizer(args)
    if dataset == 'LLama-SFT':
        train_dataset = LLamaRawDataset(args.train_data_path, args.max_padding_length)
        valid_dataset = LLamaRawDataset(args.valid_data_path, args.max_padding_length)

        return train_dataset, valid_dataset

    elif dataset == 'LLava-SFT':
        train_dataset = LLavaSupervisedDataset(args.train_data_path)
        valid_dataset = LLavaSupervisedDataset(args.valid_data_path)

        return train_dataset, valid_dataset

    elif dataset == 'Qwen-VL-SFT':
        train_dataset = QwenVLSupervisedDataset(args.train_data_path)
        valid_dataset = QwenVLSupervisedDataset(args.valid_data_path)

        return train_dataset, valid_dataset

    elif dataset == 'ChatGLM-SFT':
        train_dataset = ChatGLMRawDataset(args.train_data_path, args.source_seq_len, args.target_seq_len)
        valid_dataset = ChatGLMRawDataset(args.valid_data_path, args.source_seq_len, args.target_seq_len)
        return train_dataset, valid_dataset

    elif dataset == 'Bloom-SFT':
        train_dataset = BloomRawDataset(args.train_data_path, args.max_padding_length)
        valid_dataset = BloomRawDataset(args.valid_data_path, args.max_padding_length)
        return train_dataset, valid_dataset

    elif dataset == 'Starcoder-SFT':
        train_dataset = StarcoderRawDataset(args.train_data_path, args.max_padding_length)
        valid_dataset = StarcoderRawDataset(args.valid_data_path, args.max_padding_length)
        return train_dataset, valid_dataset

    else:
        raise NotImplementedError('dataset {} is not implemented.'.format(dataset))

def build_pretrain_dataset_from_original(dataset):

    args = get_args()
    build_tokenizer(args)
    if dataset == 'LLama-Pretrain-Raw':
        train_dataset = LLamaRawDataset(args.train_data_path, args.max_padding_length)
        #valid_dataset = LLamaRawDataset(args.valid_data_path, args.max_padding_length)
        #test_dataset = LLamaRawDataset(args.test_data_path, args.max_padding_length)
        # customize your validation and test dataset here

        return train_dataset, train_dataset, train_dataset

    elif dataset == 'LLava-Pretrain-Raw':
        train_dataset = LLavaSupervisedDataset(args.train_data_path)
        #valid_dataset = LLavaSupervisedDataset(args.valid_data_path)
        #test_dataset = LLavaSupervisedDataset(args.test_data_path)

        return train_dataset, train_dataset, train_dataset

    elif dataset == 'ChatGLM-Pretrain-Raw':
        train_dataset = ChatGLMRawDataset(args.train_data_path, args.source_seq_len, args.target_seq_len)
        #valid_dataset = ChatGLMRawDataset(args.train_data_path, args.source_seq_len, args.target_seq_len)
        #test_dataset = ChatGLMRawDataset(args.train_data_path, args.source_seq_len, args.target_seq_len)

        return train_dataset, train_dataset, train_dataset

    elif dataset == 'Starcoder-Pretrain-Raw':
        train_dataset = StarcoderRawDataset(args.train_data_path, args.max_padding_length)
        #valid_dataset = StarcoderRawDataset(args.train_data_path, args.max_padding_length)
        #test_dataset = StarcoderRawDataset(args.train_data_path, args.max_padding_length)

        return train_dataset, train_dataset, train_dataset

    else:
        raise NotImplementedError('dataset {} is not implemented.'.format(dataset))


def build_pretrain_dataset_from_idxmap(data_prefix,
                                              max_padding_length,
                                              dataset_type,
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
                                                dataset_type, splits_string,
                                                train_valid_test_num_samples,
                                                seed, skip_warmup, return_doc_ids)

    # Blending dataset.
    # Parse the values.
    from megatron.data.dataset_utils import get_datasets_weights_and_num_samples
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
            prefixes[i], max_padding_length, dataset_type, splits_string,
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
    from megatron.data.blendable_dataset import BlendableDataset
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

def _build_train_valid_test_datasets(data_prefix, max_padding_length, dataset_type, splits_string,
                                     train_valid_test_num_samples,
                                     seed, skip_warmup,
                                     return_doc_ids=False):
    try:
        from megatron.data.gpt_dataset import get_indexed_dataset_
        from megatron.data.gpt_dataset import get_train_valid_test_split_
    except:
        from megatron.data.dataset_utils import get_indexed_dataset_
        from megatron.data.dataset_utils import get_train_valid_test_split_
    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix, skip_warmup)
    try:
        total_num_of_documents = indexed_dataset.sizes.shape[0]
    except:
        total_num_of_documents = indexed_dataset.document_indices.shape[0] - 1
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)
    # Print stats about the splits.

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index],
                                  stop=splits[index + 1],
                                  step=1,
                                  dtype=np.int32)
            if dataset_type == 'LLama-Pretrain-Idxmap':
                from .llama import LLamaIdxMapDataset
                dataset = LLamaIdxMapDataset(
                    name, data_prefix, documents, indexed_dataset,
                    train_valid_test_num_samples[index],
                    seed, max_padding_length, return_doc_ids)
            else:
                raise RuntimeError("The provided dataset_type is not supported in Pretrain mode. \nChoose from [LLama-Pretrain-Idxmap].")

        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return (train_dataset, valid_dataset, test_dataset)
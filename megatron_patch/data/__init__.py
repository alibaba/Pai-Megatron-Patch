# Copyright (c) 2025 Alibaba PAI Team.
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
    # <= 240126
    from megatron import (
        get_args, 
        print_rank_0,
    )
except:
    # >= 240405
    from megatron.training import (
        get_args,
        print_rank_0
    )

from megatron.core import mpu
from megatron_patch.tokenizer import build_tokenizer, get_tokenizer
from .json_sft import JSONSFTDataset

def build_evaluation_dataset(dataset):
    raise NotImplementedError(f"Dataset {dataset} is no longer supported in Pai-Megatron-Patch anymore, downgrade to v0.10.2 or lower to use it.")

def build_finetune_dataset(dataset):
    raise NotImplementedError(f"Dataset {dataset} is no longer supported in Pai-Megatron-Patch anymore, downgrade to v0.10.2 or lower to use it.")

# NOTE: better to use shared train_valid_test_datasets_provider instead of build_dataset
def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()
    if get_tokenizer() is None:
        build_tokenizer(args)
    print_rank_0("> building train, validation, and test datasets for GPT ...")
    return build_dataset(args, train_val_test_num_samples)

def core_gpt_dataset_config_from_args(args):
    """
        NOTE: require >= 240405
    """
    from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
    from megatron.core.datasets.utils import get_blend_from_list
    tokenizer = get_tokenizer()
    kwargs =dict(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path),
        ],
        split=args.split,
        path_to_cache=args.data_cache_path,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )
    try:
        return GPTDatasetConfig(
            num_dataset_builder_threads=args.num_dataset_builder_threads, 
            **kwargs
        )
    except Exception:
        # 240405
        return GPTDatasetConfig(**kwargs)

def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def is_dataset_built_on_rank_packing():
    return mpu.get_tensor_model_parallel_rank() == 0


def build_dataset(args, train_val_test_num_samples):
    from megatron.core.datasets.gpt_dataset import (
        GPTDataset,
        MockGPTDataset,
    )
    from megatron.core.datasets.blended_megatron_dataset_builder import (
        BlendedMegatronDatasetBuilder,
    )
    if get_tokenizer() is None:
        build_tokenizer(args)
    if args.dataset == 'JSON-SFT':
        train_dataset = JSONSFTDataset(args.train_data_path, args.max_padding_length)
        val_dataset = JSONSFTDataset(args.valid_data_path, args.max_padding_length)
        test_dataset = JSONSFTDataset(args.valid_data_path, args.max_padding_length)
    elif args.dataset == 'MMAP':
        config = core_gpt_dataset_config_from_args(args)
        dataset_type = MockGPTDataset if config.mock else GPTDataset
        should_build_dataset = is_dataset_built_on_rank
        if args.train_mode != "pretrain":
            # NOTE: in data preparation scripts, the sequence is collect into (seq, labels)
            # therefore we need to double the seqlen
            config.sequence_length = config.sequence_length * 2
            if args.reset_position_ids:
                should_build_dataset = is_dataset_built_on_rank_packing

        train_dataset, val_dataset, test_dataset = BlendedMegatronDatasetBuilder(
            dataset_type, train_val_test_num_samples, should_build_dataset, config
        ).build()
        print_rank_0("> finished creating GPT datasets ...")
    else:
        raise NotImplementedError(f"Dataset {args.dataset} is no longer supported in Pai-Megatron-Patch anymore, downgrade to v0.10.2 or lower to use it.")
    
    return train_dataset, val_dataset, test_dataset


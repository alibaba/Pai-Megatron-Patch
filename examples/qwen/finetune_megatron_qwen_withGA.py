# Copyright (c) 2023 Alibaba PAI and Nvidia Megatron-LM Team.
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

from functools import partial
import torch
import os

from megatron.core.enums import ModelType
from megatron.arguments import core_transformer_config_from_args
from megatron import get_args
from megatron.core import tensor_parallel
from megatron.utils import average_losses_across_data_parallel_group

from megatron_patch.data.utils import get_batch_on_this_tp_rank_original
from megatron_patch.data import \
    build_pretrain_dataset_from_original, build_pretrain_dataset_from_idxmap
from megatron_patch.model.qwen.gpt_model import GPTModel
from megatron_patch.tokenizer import get_tokenizer, build_tokenizer
from megatron_patch.training import pretrain
from megatron_patch.arguments import get_patch_args

def model_provider(pre_process=True, post_process=True):
    args = get_args()
    build_tokenizer(args)
    config = core_transformer_config_from_args(get_args())
    model = GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=args.enable_parallel_output,
        pre_process=pre_process,
        post_process=post_process
    )
    return model

def forward_step(data_iterator, model):
    args = get_args()
    batch = get_batch_on_this_tp_rank_original(data_iterator)
    tokens = batch['tokens']
    labels = batch['labels']
    position_ids = batch["position_ids"]
    attention_mask = batch["attention_mask"]
    loss_mask = batch['loss_mask']

    logits = model(input_ids=tokens,
                   position_ids=position_ids,
                   attention_mask=attention_mask)

    if args.enable_parallel_output:

        def loss_func(loss_mask, logits):
            losses = tensor_parallel.vocab_parallel_cross_entropy(
                logits.contiguous().float(), labels.contiguous())
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
            averaged_loss = average_losses_across_data_parallel_group([loss])
            return loss, {'lm loss': averaged_loss[0]}
    else:

        def loss_func(loss_mask, logits):
            loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_func(torch.squeeze(logits).contiguous().float(), torch.squeeze(labels))
            averaged_loss = average_losses_across_data_parallel_group([loss])
            return loss, {'lm loss': averaged_loss[0]}

    return logits, partial(loss_func, loss_mask)

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    if os.path.isfile(args.train_data_path[0]):
        train_ds, valid_ds, test_ds = \
            build_pretrain_dataset_from_original(args.dataset)
    else:
        train_ds, valid_ds, test_ds = \
            build_pretrain_dataset_from_idxmap(
                data_prefix=args.train_data_path,
                max_padding_length=args.max_padding_length,
                dataset_type=args.dataset,
                splits_string=args.split,
                train_valid_test_num_samples=train_val_test_num_samples,
                seed=args.seed,
                skip_warmup=(not args.mmap_warmup)
            )

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             extra_args_provider=get_patch_args)

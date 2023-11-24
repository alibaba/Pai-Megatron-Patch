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
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.arguments import core_transformer_config_from_args
from megatron import get_args
from megatron import get_timers
from megatron.core import tensor_parallel
from megatron.utils import average_losses_across_data_parallel_group

from megatron_patch.data import build_finetune_dataset
from megatron_patch.model.llama2.gpt_model import GPTModel
from megatron_patch.tokenizer import get_tokenizer, build_tokenizer
from megatron_patch.training import pretrain
from megatron_patch.arguments import get_tasks_args

def model_provider(pre_process=True, post_process=True):
    args = get_args()
    build_tokenizer(args)
    config = core_transformer_config_from_args(get_args())
    model = GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model

def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()
    datatype = torch.int64

    keys = ['input_ids', 'labels']
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    tokens_ = data_b['input_ids'].long()
    labels = data_b['labels'].long()

    # labels = labels[:, 1:].contiguous()
    # tokens = tokens_[:, :-1].contiguous()
    tokens = tokens_.long().cuda().contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        labels,
        tokenizer.pad_token_id,
        args.reset_position_ids,
        args.reset_attention_mask,
        True)

    return tokens, labels, loss_mask, attention_mask, position_ids

def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    logits = model(tokens, position_ids, attention_mask)

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_mask = loss_mask[..., 1:].contiguous()

    def loss_func(loss_mask, shift_logits):
        losses = tensor_parallel.vocab_parallel_cross_entropy(
            shift_logits.contiguous().float(), shift_labels.contiguous())
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        averaged_loss = average_losses_across_data_parallel_group([loss])
        return loss, {'lm loss': averaged_loss[0]}

    return shift_logits, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    train_ds, valid_ds = \
        build_finetune_dataset(args.dataset)

    return train_ds, valid_ds, valid_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             extra_args_provider=get_tasks_args)

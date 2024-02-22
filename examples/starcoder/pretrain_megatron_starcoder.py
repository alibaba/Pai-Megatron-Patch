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
from megatron import get_args
from megatron.core import tensor_parallel
from megatron.utils import average_losses_across_data_parallel_group
from megatron.arguments import core_transformer_config_from_args
from megatron.utils import get_ltor_masks_and_position_ids

from megatron_patch.data import \
    build_pretrain_dataset_from_original, build_pretrain_dataset_from_idxmap

from megatron_patch.model.starcoder.gpt_model import GPTModel
from megatron_patch.tokenizer import build_tokenizer
from megatron_patch.tokenizer import get_tokenizer
from megatron_patch.training import pretrain
from megatron_patch.arguments import get_patch_args

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

def forward_step(data_iterator, model):
    tokenizer = get_tokenizer()
    keys = ['input_ids', 'labels', 'loss_mask']
    datatype = torch.int64
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['input_ids'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    args = get_args()
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens, tokenizer.eod, args.reset_position_ids,
        args.reset_attention_mask, args.eod_mask_loss)

    output_tensor = model(input_ids=tokens,
                          attention_mask=attention_mask,
                          labels=labels,
                          position_ids=position_ids)

    def loss_func(loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        averaged_loss = average_losses_across_data_parallel_group([loss])
        return loss, {'lm loss': averaged_loss[0]}

    return output_tensor, partial(loss_func, loss_mask)


if __name__ == '__main__':
    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             extra_args_provider=get_patch_args)

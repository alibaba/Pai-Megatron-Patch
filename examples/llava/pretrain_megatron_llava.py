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

from megatron.core.enums import ModelType
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.arguments import core_transformer_config_from_args
from megatron import get_args
from megatron import get_timers
from megatron.core import tensor_parallel
from megatron.utils import average_losses_across_data_parallel_group

from megatron_patch.data import build_pretrain_dataset_from_original
from megatron_patch.model.llava.gpt_model import GPTModel
from megatron_patch.tokenizer import build_tokenizer
from megatron_patch.tokenizer import get_tokenizer
from megatron_patch.training import pretrain
from megatron_patch.arguments import get_patch_args
from megatron_patch.data.llava.constants import IGNORE_INDEX

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

    text_keys = ['input_ids', 'labels']
    img_keys = ['image']
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_text = {'input_ids': data['input_ids'], 'labels': data['labels']}
    data_image = {'image': data['image']}
    data_text = tensor_parallel.broadcast_data(text_keys, data_text, torch.int64)
    data_image = tensor_parallel.broadcast_data(img_keys, data_image, torch.bfloat16)
    tokens_ = data_text['input_ids'].long()
    labels_ = data_text['labels'].long()
    images = data_image['image']
    tokens = tokens_[:, :-1].contiguous()
    labels = labels_[:, 1:].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        IGNORE_INDEX,
        args.reset_position_ids,
        args.reset_attention_mask,
        True)
    return tokens, labels, loss_mask, attention_mask, position_ids, images

def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()
    args = get_args()
    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids, images = get_batch(
        data_iterator)
    num_patch = int((args.image_size / args.patch_size) ** 2)
    timers('batch-generator').stop()
    image_label = torch.full((labels.shape[0], num_patch-1), IGNORE_INDEX, device=labels.device, dtype=labels.dtype)
    image_loss_mask = torch.zeros((labels.shape[0], num_patch-1), dtype=torch.float, device=labels.device)
    total_label = torch.cat([image_label, labels], dim=1)
    total_loss_mask = torch.cat([image_loss_mask, loss_mask], dim=1)
    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=total_label, images=images)

    return output_tensor, partial(loss_func, total_loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    train_ds, valid_ds, test_ds = \
        build_pretrain_dataset_from_original(args.dataset)

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             extra_args_provider=get_patch_args)

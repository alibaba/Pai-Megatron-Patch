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

from megatron import get_args
from megatron.core import parallel_state, tensor_parallel
from megatron.initialize import initialize_megatron
from megatron.utils import average_losses_across_data_parallel_group
from megatron.utils import get_ltor_masks_and_position_ids
from megatron_patch.finetune_utils import finetune
from megatron_patch.model.llava.gpt_model import GPTModel
from megatron_patch.tokenizer import get_tokenizer
from megatron_patch.arguments import get_patch_args
from megatron_patch.tokenizer import build_tokenizer
from megatron.arguments import core_transformer_config_from_args
from megatron_patch.data.llava.constants import IGNORE_INDEX
from megatron_patch.data import build_finetune_dataset


def model_provider(pre_process=True, post_process=True):

    config = core_transformer_config_from_args(get_args())
    model = GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model

def train_valid_test_datasets_provider():
    """Build train, valid, and test datasets."""
    args = get_args()
    train_ds, valid_ds = build_finetune_dataset(args.dataset)

    return train_ds, valid_ds


def forward_step(data_iterator, model):
    args = get_args()
    tokenizer = get_tokenizer()
    try:
        data_iterator = next(data_iterator)
    except BaseException:
        data_iterator = data_iterator

    text_keys = ['input_ids', 'labels']
    img_keys = ['image']

    data_text = {'input_ids': data_iterator['input_ids'], 'labels': data_iterator['labels']}
    data_image = {'image': data_iterator['image']}
    data_text = tensor_parallel.broadcast_data(text_keys, data_text, torch.int64)
    data_image = tensor_parallel.broadcast_data(img_keys, data_image, torch.bfloat16)
    tokens = data_text['input_ids'].long()
    labels = data_text['labels'].long()
    images = data_image['image']

    # Get the masks and postition ids.
    _, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        IGNORE_INDEX,
        args.reset_position_ids,
        args.reset_attention_mask,
        True)

    num_patch = int((args.image_size / args.patch_size) ** 2)
    image_label = torch.full((labels.shape[0], num_patch-1), IGNORE_INDEX, device=labels.device, dtype=labels.dtype)
    image_loss_mask = torch.zeros((labels.shape[0], num_patch-1), dtype=torch.float, device=labels.device)
    total_labels = torch.cat([image_label, labels], dim=1)
    attention_mask = total_labels.ne(tokenizer.pad_token_id)
    total_loss_mask = torch.cat([image_loss_mask, loss_mask], dim=1)
    logits = model(tokens, position_ids, attention_mask, images=images)

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = total_labels[..., 1:].contiguous()
    loss_mask = total_loss_mask[..., 1:].contiguous()
    def loss_func(loss_mask, shift_logits):
        losses = tensor_parallel.vocab_parallel_cross_entropy(
            shift_logits.contiguous().float(), shift_labels.contiguous())
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        averaged_loss = average_losses_across_data_parallel_group([loss])
        return loss, {'vlm loss': averaged_loss[0]}

    return shift_logits, partial(loss_func, loss_mask)

if __name__ == '__main__':

    initialize_megatron(extra_args_provider=get_patch_args)

    args = get_args()
    build_tokenizer(args)

    finetune(train_valid_datasets_provider=train_valid_test_datasets_provider,
             model_provider=model_provider,
             forward_step=forward_step)

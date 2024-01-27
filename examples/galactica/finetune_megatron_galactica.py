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
from megatron.initialize import initialize_megatron
from megatron.utils import average_losses_across_data_parallel_group

from megatron_patch.data.finetune_dataset import LLamaDataset
from megatron_patch.finetune_utils import finetune
from megatron_patch.model.galactica.gpt_model import GPTModel
from megatron_patch.tokenizer import build_tokenizer
from megatron_patch.tokenizer import get_tokenizer
from megatron_patch.arguments import get_patch_args

def model_provider(pre_process=True, post_process=True):
    model = GPTModel(num_tokentypes=0,
                     parallel_output=True,
                     pre_process=pre_process,
                     post_process=post_process)
    return model


def train_valid_datasets_provider():
    args = get_args()
    tokenizer = build_tokenizer(args)
    train_dataset = LLamaDataset(args.train_data, tokenizer,
                                 args.max_padding_length)
    valid_dataset = LLamaDataset(args.valid_data, tokenizer,
                                 args.max_padding_length)
    return train_dataset, valid_dataset


def forward_step(data_iterator, model):
    tokenizer = get_tokenizer()

    try:
        data_iterator = next(data_iterator)
    except BaseException:
        data_iterator = data_iterator

    input_ids = data_iterator['input_ids'].long().cuda().contiguous()
    labels = data_iterator['labels'].long().cuda().contiguous()
    loss_mask = data_iterator['loss_mask'].long().cuda()
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    loss_mask = loss_mask[..., 1:].contiguous()

    output_tensor = model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels)

    def loss_func(loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        averaged_loss = average_losses_across_data_parallel_group([loss])
        return loss, {'lm loss': averaged_loss[0]}

    return output_tensor, partial(loss_func, loss_mask)


if __name__ == '__main__':

    initialize_megatron(extra_args_provider=get_patch_args)

    finetune(train_valid_datasets_provider=train_valid_datasets_provider,
             model_provider=model_provider,
             forward_step=forward_step)

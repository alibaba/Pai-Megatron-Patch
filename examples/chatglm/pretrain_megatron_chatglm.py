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
from megatron import get_args
from megatron.utils import average_losses_across_data_parallel_group

from megatron_patch.data.finetune_dataset import ChatGLMDataset
from megatron_patch.model.chatglm.gpt_model import GPTModel
from megatron_patch.tokenizer import build_tokenizer
from megatron_patch.tokenizer import get_tokenizer
from megatron_patch.training import pretrain
from megatron_patch.arguments import get_tasks_args

def model_provider(pre_process=True, post_process=True):
    args = get_args()
    build_tokenizer(args)
    model = GPTModel(num_tokentypes=0,
                     parallel_output=True,
                     pre_process=pre_process,
                     post_process=post_process)
    return model


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train and validation dataset."""
    args = get_args()
    tokenizer = get_tokenizer()
    train_dataset = ChatGLMDataset(args.train_data, tokenizer,
                                   args.source_seq_len, args.target_seq_len)
    valid_dataset = ChatGLMDataset(args.valid_data, tokenizer,
                                   args.source_seq_len, args.target_seq_len)
    test_dataset = ChatGLMDataset(args.valid_data, tokenizer,
                                  args.source_seq_len, args.target_seq_len)
    return train_dataset, valid_dataset, test_dataset


def forward_step(data_iterator, model):
    """Forward step."""

    try:
        data_iterator = next(data_iterator)
    except BaseException:
        data_iterator = data_iterator
    input_ids = data_iterator['input_ids'].long().cuda()
    labels = data_iterator['labels'].long().cuda()
    lm_logits = model(input_ids=input_ids)

    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def loss_func(shift_logits):
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))
        averaged_loss = average_losses_across_data_parallel_group([loss])
        return loss, {'lm loss': averaged_loss[0]}

    return shift_logits, partial(loss_func)

if __name__ == '__main__':
    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             extra_args_provider=get_tasks_args)

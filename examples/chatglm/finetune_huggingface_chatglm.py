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

from transformers import AutoModel

from megatron import get_args
from megatron.initialize import initialize_megatron

from megatron_patch.data.finetune_dataset import ChatGLMDataset
from megatron_patch.finetune_utils import finetune
from megatron_patch.tokenizer import build_tokenizer
from megatron_patch.arguments import get_patch_args

def model_provider(pre_process=True, post_process=True):
    args = get_args()
    model = AutoModel.from_pretrained(args.load, trust_remote_code=True)
    return model


def train_valid_datasets_provider():
    """Build train and validation dataset."""
    args = get_args()
    tokenizer = build_tokenizer(args)
    train_dataset = ChatGLMDataset(args.train_data, tokenizer,
                                   args.source_seq_len, args.target_seq_len)
    valid_dataset = ChatGLMDataset(args.valid_data, tokenizer,
                                   args.source_seq_len, args.target_seq_len)
    return train_dataset, valid_dataset


def forward_step(data_iterator, model):
    try:
        data_iterator = next(data_iterator)
    except BaseException:
        data_iterator = data_iterator

    tokens = data_iterator['input_ids'].long().cuda()
    # huggingface will shift labels inside transformers
    labels = data_iterator['labels'].long().cuda()
    output_tensor = model(input_ids=tokens, labels=labels)
    return output_tensor.loss


if __name__ == '__main__':
    initialize_megatron(extra_args_provider=get_patch_args)

    finetune(train_valid_datasets_provider=train_valid_datasets_provider,
             model_provider=model_provider,
             forward_step=forward_step)

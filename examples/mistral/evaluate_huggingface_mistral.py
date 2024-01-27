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

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron.core.enums import ModelType
from megatron import get_args
from megatron import print_rank_0
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.p2p_communication import send_forward
from megatron.initialize import initialize_megatron
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module
from megatron.utils import unwrap_model
from megatron.arguments import core_transformer_config_from_args

from megatron_patch.data import build_evaluation_dataset
from megatron_patch.finetune_utils import build_data_loader
from megatron_patch.tokenizer import get_tokenizer
from megatron_patch.training import get_model
from megatron_patch.arguments import get_patch_args
from transformers import AutoModelForCausalLM

def get_model_provider():
    """Based on evaluation metric set the parallel-output flag and
    return the model provider."""
    def model_provider(pre_process=True, post_process=True):
        args = get_args()
        model = AutoModelForCausalLM.from_pretrained(args.load, device_map="auto")
        return model

    return model_provider


def forward_step(batch, model):
    """Forward step."""
    tokenizer = get_tokenizer()
    # Get the batch.
    input_ids = batch['input_ids'].long().cuda()
    labels = batch['labels'].long().cuda()
    labels[labels == 0] = -100
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    # Tell the model what our actual batch size will be
    args = get_args()

    # Forward pass through the model.
    unwrapped_model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module))
    output = unwrapped_model(input_ids=input_ids,
                             labels=labels,
                             attention_mask=attention_mask)
    config = core_transformer_config_from_args(args)
    send_forward(output, config)
    if parallel_state.is_pipeline_last_stage():
        print_rank_0(output.loss)
        return output.loss

    return None


def evaluate(data_loader, model):
    """Evaluation."""
    args = get_args()

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_output = 0.0
    with torch.no_grad():
        # For all the batches in the dataset.
        for iteration, batch in enumerate(data_loader):
            if iteration % args.log_interval == 0:
                print_rank_0('> working on iteration: {}'.format(iteration))
            # Forward evaluation.
            output = forward_step(batch, model)

            # Reduce across processes.
            if parallel_state.is_pipeline_last_stage():
                torch.distributed.all_reduce(
                    output, group=parallel_state.get_data_parallel_group())

                total_output += output

    return total_output

def main():
    """Main program."""
    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print('Interleaved pipeline schedule '
              'is not yet supported for text generation.')
        exit()

    # Set up model and load checkpoint.
    model = get_model(get_model_provider(),
                      model_type=ModelType.encoder_or_decoder,
                      wrap_with_ddp=False)

    assert len(model) == 1, 'Above condition should have caught this'
    model = model[0]

    # Data stuff.
    dataset = build_evaluation_dataset(args.dataset)
    dataloader = build_data_loader(dataset,
                                   args.micro_batch_size,
                                   args.num_workers,
                                   drop_last=False)

    # Run evaluation.
    evaluate(dataloader, model)
    print_rank_0('done :-)')


if __name__ == '__main__':
    initialize_megatron(extra_args_provider=get_patch_args)
    main()

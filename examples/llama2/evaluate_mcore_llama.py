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
from typing import Union
from megatron.core.enums import ModelType
import megatron.model
from megatron import get_args
from megatron import print_rank_0
from megatron.core import parallel_state, tensor_parallel
from megatron.core.pipeline_parallel.p2p_communication import recv_forward
from megatron.core.pipeline_parallel.p2p_communication import send_forward
from megatron.initialize import initialize_megatron
from megatron.utils import unwrap_model
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.arguments import core_transformer_config_from_args
from megatron.checkpointing import load_checkpoint

from megatron_patch.data import build_evaluation_dataset
from megatron_patch.finetune_utils import build_data_loader
from megatron_patch.arguments import get_patch_args
from megatron_patch.tokenizer import get_tokenizer
from megatron_patch.training import get_model
from megatron_patch.model.mixtral_bak.layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt import GPTModel

import torch._dynamo

torch._dynamo.config.suppress_errors = True

def get_model_provider():
    def model_provider(
        pre_process=True, post_process=True
    ) -> Union[GPTModel, megatron.model.GPTModel]:
        args = get_args()
        config = core_transformer_config_from_args(get_args())
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm)
        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
        )

        return model

    return model_provider

def get_batch(batch):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['input_ids', 'labels']
    datatype = torch.int64

    data_b = tensor_parallel.broadcast_data(keys, batch, datatype)

    tokens = data_b['input_ids'].long().cuda().contiguous()
    labels = data_b['labels'].long().cuda().contiguous()

    tokens = tokens[:, :-1].contiguous()
    labels = labels[:, 1:].contiguous()
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        labels,
        tokenizer.pad_token_id,
        args.reset_position_ids,
        args.reset_attention_mask,
        True)

    return tokens, labels, loss_mask, attention_mask, position_ids

def forward_step(batch, model):
    """Forward step."""

    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        batch)

    # Tell the model what our actual batch size will be
    args = get_args()
    args.micro_batch_size = len(labels)
    config = core_transformer_config_from_args(args)
    tensor_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)
    input_tensor = recv_forward(tensor_shape, config)

    # Forward pass through the model.
    unwrapped_model = unwrap_model(model)
    unwrapped_model.set_input_tensor(input_tensor)
    output = model(tokens, position_ids, attention_mask)
    send_forward(output, config)
    #if parallel_state.is_pipeline_last_stage():
    if output.shape[-1] != args.hidden_size:
        loss_mask = loss_mask.view(-1).float()
        # For loss, return the unreduced loss.
        losses = tensor_parallel.vocab_parallel_cross_entropy(
            output.contiguous().float(), labels.contiguous())
        loss = torch.sum(
            losses.view(-1) * loss_mask.contiguous().view(-1).float()) / loss_mask.sum()
        print(loss)
        print_rank_0(loss)
        return loss

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

    # Data stuff.
    dataset = build_evaluation_dataset(args.dataset)
    dataloader = build_data_loader(dataset,
                                   args.micro_batch_size,
                                   args.num_workers,
                                   drop_last=False)


    # Set up model and load checkpoint.
    model = get_model(get_model_provider(),
                      model_type=ModelType.encoder_or_decoder,
                      wrap_with_ddp=False)

    if args.load is not None:
        load_checkpoint(model, None, None)

    assert len(model) == 1, 'Above condition should have caught this'
    model = model[0]



    # Run evaluation.
    evaluate(dataloader, model)
    print_rank_0('done :-)')


if __name__ == '__main__':
    initialize_megatron(extra_args_provider=get_patch_args)
    main()

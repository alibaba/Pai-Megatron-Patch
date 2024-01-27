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

import math
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from sat.model import GLM130B as GPTModel
from sat.training import load_checkpoint

from megatron import get_args
from megatron import print_rank_0
from megatron import is_last_rank
from megatron.core import parallel_state, tensor_parallel
from megatron.core.pipeline_parallel.p2p_communication import send_forward
from megatron.initialize import initialize_megatron
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module
from megatron.utils import unwrap_model

from megatron_patch.data.evaluate_dataset import build_evaluation_dataset
from megatron_patch.finetune_utils import build_data_loader
from megatron_patch.tokenizer import build_tokenizer
from megatron_patch.training import get_model
from megatron_patch.arguments import get_patch_args


def get_model_provider(eval_metric):
    """Based on evaluation metric set the parallel-output flag and
    return the model provider."""
    def model_provider(pre_process=True, post_process=True):
        """Build the model."""
        args = get_args()
        print_rank_0('building GPT model ...')
        build_tokenizer(args)
        args.model_parallel_size = args.tensor_model_parallel_size
        args.vocab_size = 150528
        args.max_sequence_length = args.seq_length
        args.layernorm_order = 'post'
        args.skip_init = True
        args.inner_hidden_size = 32768
        args.position_encoding_2d = False
        args.no_glu = False
        model = GPTModel(args)

        return model

    return model_provider


def process_batch(batch):
    """Process batch and produce inputs for the model."""

    tokens = batch['tokens'].long().cuda().contiguous()
    labels = batch['targets'].long().cuda().contiguous()
    attention_mask = batch['attention_mask'].long().cuda().contiguous()
    loss_mask = batch['loss_mask'].long().cuda().contiguous()
    position_ids = batch['position_ids'].long().cuda().contiguous()
    attention_mask = attention_mask.to(torch.bool).unsqueeze(1)

    return tokens, labels, attention_mask, position_ids, loss_mask


def forward_step(batch, model, eval_metric):
    """Forward step."""

    # Get the batch.
    tokens, labels, attention_mask, position_ids, loss_mask = process_batch(
        batch)

    # Tell the model what our actual batch size will be
    args = get_args()
    args.micro_batch_size = len(labels)

    # Forward pass through the model.
    unwrapped_model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module))
    original_parallel_output = unwrapped_model.transformer.parallel_output
    unwrapped_model.transformer.parallel_output = True
    output, *output_per_layers = unwrapped_model(tokens, position_ids,
                                                 attention_mask)
    unwrapped_model.transformer.parallel_output = original_parallel_output
    send_forward(output)
    if parallel_state.is_pipeline_last_stage():
        # For loss, return the unreduced loss.
        if eval_metric == 'loss':
            losses = tensor_parallel.vocab_parallel_cross_entropy(
                output.contiguous().float(), labels.contiguous())
            loss = torch.sum(
                losses.view(-1) * loss_mask.contiguous().view(-1).float())
            return loss

        # For accuracy, return the number of correctly predicted samples.
        if eval_metric == 'accuracy':
            outputs = torch.argmax(output, -1)
            correct = (outputs == labels).float()
            correct[(1 - loss_mask).bool()] = 1
            correct = correct.prod(-1)
            return correct.sum()

        raise NotImplementedError('forward method for evaluation metric {} '
                                  'is not implemented.'.format(eval_metric))
    return None


def evaluate(data_loader, model, eval_metric):
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
            output = forward_step(batch, model, eval_metric)

            # Reduce across processes.
            if parallel_state.is_pipeline_last_stage():
                torch.distributed.all_reduce(
                    output, group=parallel_state.get_data_parallel_group())

                total_output += output

    return total_output


def evaluate_and_print_results(task, data_loader, model, eval_metric):
    """Evaluate and print results on screen."""

    # Evaluate and get results.
    output = evaluate(data_loader, model, eval_metric)

    string = ' validation results on {} | '.format(task)
    if is_last_rank():
        if eval_metric == 'loss':
            num_tokenized_tokens = data_loader.dataset.num_tokenized_tokens
            num_original_tokens = data_loader.dataset.num_original_tokens
            val_loss = output / (num_tokenized_tokens - 1)
            ppl = math.exp(min(20, val_loss))
            token_ratio = (num_tokenized_tokens - 1) / (num_original_tokens -
                                                        1)
            adjusted_ppl = math.exp(min(20, val_loss * token_ratio))
            string += 'avg loss: {:.4E} | '.format(val_loss)
            string += 'ppl: {:.4E} | '.format(ppl)
            string += 'adjusted ppl: {:.4E} | '.format(adjusted_ppl)
            string += 'token ratio: {} |'.format(token_ratio)

        elif eval_metric == 'accuracy':
            num_examples = len(data_loader.dataset)
            acc = output / num_examples
            string += 'number correct: {:.4E} | '.format(output)
            string += 'total examples: {:.4E} | '.format(num_examples)
            string += 'avg accuracy: {:.4E}'.format(acc)

        else:
            raise NotImplementedError('evaluation method for {} metric is not '
                                      'implemented yet.'.format(eval_metric))

        length = len(string) + 1
        print('-' * length)
        print(string)
        print('-' * length)


def main():
    """Main program."""
    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print('Interleaved pipeline schedule'
              ' is not yet supported for text generation.')
        exit()

    if args.dataset == 'LAMBADA':
        eval_metric = 'accuracy'
    elif args.dataset == 'WIKITEXT103' or\
            args.dataset == 'GLM130B-WIKITEXT103':
        eval_metric = 'loss'
    else:
        raise NotImplementedError('{} task is not implemented.'.format(
            args.dataset))

    # Set up model and load checkpoint.
    model = get_model(get_model_provider(eval_metric), wrap_with_ddp=False)
    if args.load is not None:
        load_checkpoint(model[0], args)

    assert len(model) == 1, 'Above condition should have caught this'
    model = model[0]

    # Data stuff.
    dataset = build_evaluation_dataset(args.dataset)
    dataloader = build_data_loader(dataset,
                                   args.micro_batch_size,
                                   args.num_workers,
                                   drop_last=False)

    # Run evaluation.
    evaluate_and_print_results(args.dataset, dataloader, model, eval_metric)

    print_rank_0('done :-)')


if __name__ == '__main__':
    initialize_megatron(extra_args_provider=get_patch_args)
    main()

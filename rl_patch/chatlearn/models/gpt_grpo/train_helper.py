# Copyright 2024 Alibaba Group Holding Limited. All Rights Reserved.
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
# ==============================================================================

import copy
import torch
from functools import partial

from megatron.training import print_rank_last, is_last_rank, get_args, get_timers
from megatron.training.training import print_datetime
from megatron.training.utils import report_memory, unwrap_model
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.training.utils import average_losses_across_data_parallel_group, get_ltor_masks_and_position_ids

from chatlearn.utils import to_device

from tokenizer import get_tokenizer
from .utils import pad_to_max_len, generate_loss_mask_position_ids

def training_log(loss_dict, total_loss_dict, learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 grad_norm, params_norm, num_zeros_in_grad,
                 stats, more_grad_norm, name,
                 metric_list=None):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'
    skipped_iters_key = 'skipped iterations'
    nan_iters_key = 'nan iterations'
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(
            advanced_iters_key, 0) + 1
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(
        skipped_iters_key, 0) + skipped_iter
    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(
                key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(
        nan_iters_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = [
        'forward-backward',
        'forward-compute',
        'backward-compute',
        'batch-generator',
        'forward-recv',
        'forward-send',
        'backward-recv',
        'backward-send',
        'forward-send-forward-recv',
        'forward-send-backward-recv',
        'backward-send-forward-recv',
        'backward-send-backward-recv',
        'forward-backward-send-forward-backward-recv',
        'layernorm-grads-all-reduce',
        'embedding-grads-all-reduce',
        'grads-all-reduce',
        'grads-reduce-scatter',
        'params-all-gather',
        'optimizer-copy-to-main-grad',
        'optimizer-unscale-and-check-inf',
        'optimizer-clip-main-grad',
        'optimizer-count-zeros',
        'optimizer-inner-step',
        'optimizer-copy-main-to-model-params',
        'optimizer']

    # Calculate batch size.
    batch_size = args.micro_batch_size * args.data_parallel_size * \
        get_num_microbatches()

    total_iterations = total_loss_dict[advanced_iters_key] + \
                       total_loss_dict[skipped_iters_key]

    iter_dict = {}
    consumed_train_samples_dict = {}
    # Tensorboard values.
    if (iteration % args.tensorboard_log_interval == 0 ) and \
       is_last_rank():


        for key in loss_dict:
            iter_dict[f'{name}/{key}'] = loss_dict[key]
            consumed_train_samples_dict[f'{name}/' + key + ' vs samples'] = loss_dict[key]


        if grad_norm is not None:
            iter_dict[f'{name}/' +'grad_norm'] = grad_norm
            consumed_train_samples_dict[f'{name}/' +'grad-norm vs samples'] = grad_norm

        if more_grad_norm is not None:
            for k in more_grad_norm:
                iter_dict[f'{name}/{k}' + ' grad_norm'] = more_grad_norm[k]
                consumed_train_samples_dict[f'{name}/{k}' + ' grad-norm vs samples'] = more_grad_norm[k]


        if params_norm is not None:
            iter_dict[f'{name}/' +'params-norm'] = params_norm
            consumed_train_samples_dict[f'{name}/' +'params-norm vs samples'] = params_norm


    if iteration % args.log_interval == 0:
        elapsed_time = 0
        elapsed_time_per_iteration = elapsed_time / total_iterations
        if args.log_timers_to_tensorboard:
            iter_dict[f'{name}/' +'iteration-time'] = elapsed_time_per_iteration

        log_string = ' iteration {:8d}/{:8d} |'.format(
            iteration, args.train_iters)
        log_string += ' consumed samples: {:12d} |'.format(
            args.consumed_train_samples)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time_per_iteration * 1000.0)
        log_string += ' learning rate: {:.3E} |'.format(learning_rate)
        log_string += ' global batch size: {:5d} |'.format(batch_size)


        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key,
                           nan_iters_key]:
                avg = total_loss_dict[key].item() / \
                      float(max(1, total_loss_dict[advanced_iters_key]))
                if avg > 0.0:
                    log_string += ' {}: {:.6E} |'.format(key, avg)

                total_loss_dict[key] = torch.cuda.FloatTensor([0.0])
        log_string += ' loss scale: {:.1f} |'.format(loss_scale)

        if grad_norm is not None:
            log_string += ' grad norm: {:.3f} |'.format(grad_norm)

        if more_grad_norm is not None:
            for k in more_grad_norm:
                log_string += '{} grad norm: {:.3f} |'.format(k, more_grad_norm[k])

        log_string += ' number of nan iterations: {:3d} |'.format(
            total_loss_dict[nan_iters_key])
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        print_rank_last(log_string)
        if report_memory_flag and learning_rate > 0.:
            report_memory('(after {} iterations)'.format(iteration))
            report_memory_flag = False
        print_datetime('Logger')
        timers.log(timers_to_log, normalizer=args.log_interval)

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == (
                torch.distributed.get_world_size() - 1):
            wandb_dicts = {}
            wandb_dicts.update(stats)
            wandb_dicts.update(iter_dict)
            wandb_dict_copy = copy.deepcopy(wandb_dicts)
            if metric_list is None:
                metric_list = [wandb_dict_copy]
            else:
                metric_list.append(wandb_dict_copy)

    return report_memory_flag

def get_batch(batch_data):
    """Generate a batch"""
    args = get_args()

    data_b = next(batch_data)
    prompt_token_length = to_device("cuda", data_b["prompt_token_length"])
    response_token_length = to_device("cuda", data_b["response_token_length"])
    ref_logprobs = data_b["ref_logprobs"].float()
    old_logprobs = data_b["old_logprobs"].float()
    advantages = data_b["advantages"]
    tokens_ = data_b["all_tokens"].long()

    max_size = args.seq_length + 1
    tokens_ = pad_to_max_len(tokens_, max_size, pad_value=get_tokenizer().eod_id)

    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    loss_mask, _ = generate_loss_mask_position_ids(tokens, prompt_token_length, response_token_length)
    loss_mask = loss_mask[: , 1:]
    loss_mask = pad_to_max_len(loss_mask, args.seq_length, pad_value=0)

    # Get the masks and position ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        get_tokenizer().eod_id,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
    )

    inputs = {
        "all_tokens": tokens,
        "all_token_attention_mask": attention_mask,
        "all_token_position_ids": position_ids,
        "all_token_loss_mask": loss_mask,
        "labels": labels,
        "advantages": advantages,
        "ref_logprobs":ref_logprobs,
        "old_logprobs": old_logprobs,
    }

    for k, v in inputs.items():
        inputs[k] = to_device("cuda", v)

    return inputs

def loss_func(inputs, losses):

    ppo_losses, kl_losses, entropy_losses = losses

    loss_mask = inputs["all_token_loss_mask"]
    ppo_loss = torch.mean(ppo_losses)

    with torch.no_grad():
        num_tokens = loss_mask.sum().float()
        # reduced all losses in this microbatch
        data = average_losses_across_data_parallel_group([ppo_losses.sum(), entropy_losses.sum(), kl_losses.sum(), num_tokens])
        reduced_data = {
            "pg_loss": (data[0], data[-1]),
            "entropy_loss": (data[1], data[-1]),
            "kl_loss": (data[2], data[-1]),
        }
    return ppo_loss, reduced_data


def forward_step(data_iterator, model):
    """Forward step."""

    inputs = get_batch(data_iterator)

    output_tensor = model(input_ids=inputs["all_tokens"],
                          position_ids=inputs["all_token_position_ids"],
                          attention_mask=inputs["all_token_attention_mask"],
                          labels=inputs['labels'],
                          training_inputs=inputs)

    return output_tensor, partial(loss_func, inputs)

def inference_get_batch(data):
    """Generate a batch"""
    args = get_args()

    tokens_ = data["all_tokens"].long()

    # pad to max seq length or to tp*N
    max_size = args.seq_length + 1
    pad_all_tokens = pad_to_max_len(tokens_, max_size, pad_value=get_tokenizer().eod_id)

    labels = pad_all_tokens[:, 1:]
    tokens_ = pad_all_tokens[:, :-1]

    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens_,
        get_tokenizer().eod_id,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
    )

    inputs = {
        "all_tokens": tokens_,
        "all_token_attention_mask": attention_mask,
        "all_token_position_ids": position_ids,
        "labels": labels,

    }

    for k, v in inputs.items():
        inputs[k] = to_device("cuda", v)

    return inputs

def inference_loss_func(output_tensor, non_loss_data=True):

    return output_tensor

def inference_forward_step(data_iterator, model):
    """Forward step."""

    inputs = inference_get_batch(data_iterator)

    output_tensor = model(input_ids=inputs["all_tokens"],
                          position_ids=inputs["all_token_position_ids"],
                          attention_mask=inputs["all_token_attention_mask"],
                          labels=inputs['labels'])

    return output_tensor, partial(inference_loss_func)

# Copyright (c) 2025 Alibaba PAI and Nvidia Megatron-LM Team.
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
"""Pretrain GPT."""

import os
from functools import partial

import torch
import torch._dynamo
from megatron.core import mpu

from megatron.training import get_args, get_timers
from megatron.training.utils import (
    average_losses_across_data_parallel_group,
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
)

from megatron.core.packed_seq_params import PackedSeqParams
from megatron_patch.data.utils import (
    get_batch_on_this_tp_rank_original, 
    get_batch_on_this_tp_rank_idxmap_sft,
    _get_batch_on_this_tp_rank
)

_TORCH_MAJOR, _TORCH_MINOR = torch.__version__.split('.')[:2]
if int(_TORCH_MAJOR) >= 2 and int(_TORCH_MINOR) >= 6:
    def add_torchload_allowed_objects():
        # NOTE: since PyTorch 2.6.0, the default value of `weights_only`
        # in torch.load is changed to True.
        from argparse import Namespace
        from megatron.core.enums import ModelType
        safe_objs = [Namespace, ModelType]
        try:
            # Since 250217
            from megatron.core.transformer.enums import AttnBackend
            from megatron.core.rerun_state_machine import (
                RerunMode, 
                RerunState, 
                RerunDiagnostic
            )
            safe_objs.extend([
                AttnBackend,
                RerunMode, 
                RerunState, 
                RerunDiagnostic
            ])
        except:
            pass
        
        torch.serialization.add_safe_globals(safe_objs)
    add_torchload_allowed_objects()

def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None, None, None

    args = get_args()

    if args.dataset == 'JSON-SFT':
        if args.train_mode == "pretrain":
            raise ValueError('The JSON-SFT dataset should only be used for finetuning!')
        # get batches based on the TP rank you are on
        batch = get_batch_on_this_tp_rank_original(data_iterator, per_seq_average=True)
        # slice batch along sequence dimension for context parallelism
        num_seqs = batch.pop('num_seqs')
        batch = get_batch_on_this_cp_rank(batch)

        return (
            batch['tokens'],
            batch['labels'],
            batch['loss_mask'],
            batch['attention_mask'],
            batch['position_ids'],
            num_seqs,
            None
        )
    elif args.dataset == 'MMAP':
        # get batches based on the TP rank you are on
        if args.train_mode == "pretrain":
            if args.use_multi_token_prediction:
                batch = _get_batch_on_this_tp_rank(data_iterator)
            else:
                batch = get_batch_on_this_tp_rank(data_iterator)
        else:
            batch = get_batch_on_this_tp_rank_idxmap_sft(data_iterator, per_seq_average=True)
        
        packed_seq_params = None
        if args.reset_position_ids:
            # sequence-packing, build cu_seqlens
            position_ids = batch.get('position_ids', None)
            if position_ids is not None:
                # mbs = 1
                position_ids = position_ids[0] # shape: [seq_length]
                start_indices = (position_ids == 0).nonzero(as_tuple=True)[0]
                seqlens = start_indices[1:] - start_indices[:-1]
                # NOTE: cu_seqlens: [0, A1, A1+A2, A1+A2+A3, ..., seq_len]
                cu_seqlens = torch.zeros(start_indices.shape[0] + 1, device=position_ids.device, dtype=torch.int)
                cu_seqlens[1:-1] = torch.cumsum(seqlens, dim=0)
                cu_seqlens[-1] = position_ids.shape[0]
                packed_seq_params = PackedSeqParams(
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_kv=cu_seqlens,
                    qkv_format='thd'
                )
        
        if packed_seq_params is not None and args.context_parallel_size > 1:
            raise ValueError('Sequence Packing is not supported when CP>1 !')
        # slice batch along sequence dimension for context parallelism
        num_seqs = batch.pop('num_seqs', None)
        batch = get_batch_on_this_cp_rank(batch)

        return (
            batch['tokens'],
            batch['labels'],
            batch['loss_mask'],
            batch['attention_mask'],
            batch['position_ids'],
            num_seqs,
            packed_seq_params
        )
    else:
        raise ValueError("please set correct --dataset ")


def loss_func(loss_mask: torch.Tensor, num_seqs: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()

    loss = torch.stack([torch.sum(losses.view(-1) * loss_mask), loss_mask.sum()])
    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan().any(), (
            f"Rank {global_rank}: found NaN in local forward loss calculation. "
            f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
        )

    averaged_loss = average_losses_across_data_parallel_group(loss)
    averaged_loss = averaged_loss[0] / averaged_loss[1]

    # NOTE: The grad will be scaled down by CP size later, should not remove this multilication factor
    # LINK: https://github.com/NVIDIA/Megatron-LM/issues/906
    # The issue is solved since 0926

    if num_seqs is None:
        return loss[0] * args.context_parallel_size, {"lm loss": averaged_loss}
    return loss[0] * args.context_parallel_size, num_seqs.sum(), {"lm loss": averaged_loss}

def forward_step(data_iterator, model):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    timers = get_timers()
    args = get_args()

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids, num_seqs, packed_seq_params = get_batch(data_iterator)
    timers("batch-generator").stop()
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels, packed_seq_params=packed_seq_params)

    return output_tensor, partial(loss_func, loss_mask, num_seqs)

def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0

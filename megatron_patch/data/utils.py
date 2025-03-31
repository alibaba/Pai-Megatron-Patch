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
from megatron.core import mpu
try:
    from megatron import get_args
except:
    from megatron.training import get_args

from megatron_patch.tokenizer import get_tokenizer

def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss,
                                    create_attention_mask: bool=True):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    if create_attention_mask:
        attention_mask = torch.tril(torch.ones(
            (att_mask_batch, seq_length, seq_length), device=data.device)).view(
                att_mask_batch, 1, seq_length, seq_length)
    else:
        attention_mask = None

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask and attention_mask is not None:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    if attention_mask is not None:
        # Convert attention mask to binary:
        attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids

def get_ltor_position_ids_packed_seq(data):
    """
        Given a input_seqs from custom mmap dataset, generate a 
        position_ids by searching negative tokens.
    """

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    position_ids = position_ids.clone()

    # Loop through the batches:
    for b in range(micro_batch_size):
        # Find indecies where EOD token is.
        eod_index = position_ids[b, data[b] < 0]
        # Detach indecies from positions if going to modify positions.
        eod_index = eod_index.clone()
        # Loop through EOD indecies:
        prev_index = 0
        for j in range(eod_index.size()[0]):
            i = eod_index[j]
            position_ids[b, (i + 1):] -= (i + 1 - prev_index)
            prev_index = i + 1

    return position_ids

def get_batch_on_this_tp_rank_original(data_iterator, per_seq_average=False):
    args = get_args()
    tokenizer = get_tokenizer()
    def _broadcast(item):
        if item is None:
            return
        torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(),
                                    group=mpu.get_tensor_model_parallel_group())

    if mpu.get_tensor_model_parallel_rank() == 0:

        if isinstance(data_iterator, dict):
            data = data_iterator
        else:
            data = next(data_iterator)

        tokens_ = data['input_ids'].long()
        labels_ = data['labels'].long()
        tokens = tokens_[:, :-1].contiguous()
        labels = labels_[:, 1:].contiguous()
        # core/tensor_parallel/cross_entropy.py, target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        # labels[labels == tokenizer.eos_token_id] = -100
        # NOTE: if eos == pad, we map <eos> to  - 1 - eos_id, map these tokens back
        tokens[tokens < 0] = - 1 - tokens[tokens < 0]
        eos_indices = (labels < 0).nonzero()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[eos_indices[:, 0], eos_indices[:, 1]] = - 1 - labels[eos_indices[:, 0], eos_indices[:, 1]]
        
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            labels,
            -100,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)
        
        num_seqs = None
        if per_seq_average:
            # NOTE: raw dataset does not support sequence packing
            num_seqs = torch.ones(position_ids.shape[0], device=torch.cuda.current_device(), dtype=torch.int64)
            loss_mask = loss_mask / loss_mask.sum(dim=-1, keepdims=True) # [mbs]       

        batch = {
            'tokens': tokens.cuda(non_blocking=True),
            'labels': labels.cuda(non_blocking=True),
            'loss_mask': loss_mask.cuda(non_blocking=True),
            'attention_mask': attention_mask.cuda(non_blocking=True),
            'position_ids': position_ids.cuda(non_blocking=True),
            'num_seqs': num_seqs.cuda(non_blocking=True) if num_seqs is not None else None
        }

        if args.pipeline_model_parallel_size == 1:
            _broadcast(batch['tokens'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])
            _broadcast(batch['num_seqs'])

        elif mpu.is_pipeline_first_stage():
            _broadcast(batch['tokens'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])

        elif mpu.is_pipeline_last_stage():
            # Multi-Token Prediction (MTP) layers need tokens and position_ids to calculate embedding.
            # Currently the Multi-Token Prediction (MTP) layers is fixed on the last stage, so we need
            # to broadcast tokens and position_ids to all of the tensor parallel ranks on the last stage.
            if getattr(args, 'mtp_num_layers', None) is not None:
                _broadcast(batch['tokens'])
                _broadcast(batch['position_ids'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['num_seqs'])

    else:

        tokens = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                             device=torch.cuda.current_device())
        labels = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                             device=torch.cuda.current_device())
        loss_mask = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.float32,
                                device=torch.cuda.current_device())
        mbs = args.micro_batch_size if args.reset_attention_mask else 1
        attention_mask = torch.empty((mbs, 1, args.seq_length, args.seq_length), dtype=torch.bool,
                                     device=torch.cuda.current_device())
        position_ids = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                                   device=torch.cuda.current_device())
        
        num_seqs = None
        if per_seq_average:
            num_seqs = torch.empty((args.micro_batch_size,), dtype=torch.int64,
                                    device=torch.cuda.current_device()) 

        if args.pipeline_model_parallel_size == 1:
            _broadcast(tokens)
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(position_ids)
            _broadcast(num_seqs)

        elif mpu.is_pipeline_first_stage():
            labels = None
            loss_mask = None
            num_seqs = None

            _broadcast(tokens)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif mpu.is_pipeline_last_stage():
            # Multi-Token Prediction (MTP) layers need tokens and position_ids to calculate embedding.
            # Currently the Multi-Token Prediction (MTP) layers is fixed on the last stage, so we need
            # to broadcast tokens and position_ids to all of the tensor parallel ranks on the last stage.
            if getattr(args, 'mtp_num_layers', None) is not None:
                _broadcast(tokens)
                _broadcast(position_ids)
            else:
                tokens = None
                position_ids = None
            
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(num_seqs)

        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'num_seqs': num_seqs
        }

    return batch

def get_position_id_on_this_tp_rank_idxmap_sft_packing(data_iterator):
    args = get_args()
    tokenizer = get_tokenizer()
    def _broadcast(item):
        if item is None:
            return
        torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(),
                                    group=mpu.get_tensor_model_parallel_group())
        
    if mpu.get_tensor_model_parallel_rank() == 0:
        if isinstance(data_iterator, dict):
            data = data_iterator
        else:
            data = next(data_iterator)

        actual_seqlen = args.seq_length
        data['tokens'] = data['tokens'].long()
        tokens = data['tokens'][..., :actual_seqlen]
        position_ids = get_ltor_position_ids_packed_seq(tokens).cuda(non_blocking=True)
    else:
        # dtype: long
        position_ids = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                                   device=torch.cuda.current_device())
    _broadcast(position_ids)
    return position_ids

def get_batch_on_this_tp_rank_idxmap_sft(data_iterator, per_seq_average=False):
    args = get_args()
    tokenizer = get_tokenizer()
    def _broadcast(item):
        if item is None:
            return
        torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(),
                                    group=mpu.get_tensor_model_parallel_group())
        
    if mpu.get_tensor_model_parallel_rank() == 0:

        if isinstance(data_iterator, dict):
            data = data_iterator
        else:
            data = next(data_iterator)

        # sanity check
        assert data['tokens'].shape[-1] == 2 * args.seq_length
        actual_seqlen = args.seq_length
        data['tokens'] = data['tokens'].long()
        tokens = data['tokens'][..., :actual_seqlen]
        labels = data['tokens'][..., actual_seqlen:]
        loss_mask = (labels != -100).float()
        
        if args.reset_position_ids:
            attention_mask = None
            position_ids = get_ltor_position_ids_packed_seq(tokens)
            has_pad = tokens[:, -1] >= 0
            tokens[tokens < 0] = - tokens[tokens < 0] - 1
        else:
            tokens[tokens < 0] = - tokens[tokens < 0] - 1
            attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                tokens,
                tokenizer.eod,
                args.reset_position_ids,
                args.reset_attention_mask,
                False,
                args.create_attention_mask_in_dataloader
            )

        num_seqs = None
        if per_seq_average:
            num_seqs = torch.ones(position_ids.shape[0], device=torch.cuda.current_device(), dtype=torch.int64)
            if args.reset_position_ids:
                for b in range(position_ids.shape[0]):
                    p = position_ids[b]
                    start_indices = (p == 0).nonzero(as_tuple=True)[0]
                    seqlens = start_indices[1:] - start_indices[:-1]
                    seqlens = seqlens.cpu().numpy().tolist() + [p.shape[0] - start_indices[-1].item()]
                    subseqs = torch.split(loss_mask[b], seqlens)    
                    num_seqs[b] = len(seqlens) - int(has_pad[b])
                    for subseq_idx, (start_idx, seqlen, subseq) in enumerate(zip(start_indices, seqlens, subseqs)):
                        if subseq_idx == num_seqs[b]:
                            # NOTE: do not process pad sequence
                            continue
                        assert subseq.sum() > 0
                        loss_mask[b, start_idx: start_idx + seqlen] /= subseq.sum()
            else:
                loss_mask = loss_mask / loss_mask.sum(dim=-1, keepdims=True) # [mbs]       
                
        # dtype: long, long, float, bool, long
        batch = {
            'tokens': tokens.cuda(non_blocking=True),
            'labels': labels.cuda(non_blocking=True),
            'loss_mask': loss_mask.cuda(non_blocking=True),
            'attention_mask': attention_mask.cuda(non_blocking=True) if attention_mask is not None else None,
            'position_ids': position_ids.cuda(non_blocking=True),
            'num_seqs': num_seqs.cuda(non_blocking=True) if num_seqs is not None else None
        }

        if args.pipeline_model_parallel_size == 1:
            _broadcast(batch['tokens'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['num_seqs'])

        elif mpu.is_pipeline_first_stage():
            _broadcast(batch['tokens'])
            _broadcast(batch['attention_mask'])

        elif mpu.is_pipeline_last_stage():
            # Multi-Token Prediction (MTP) layers need tokens and position_ids to calculate embedding.
            # Currently the Multi-Token Prediction (MTP) layers is fixed on the last stage, so we need
            # to broadcast tokens and position_ids to all of the tensor parallel ranks on the last stage.
            if getattr(args, 'mtp_num_layers', None) is not None:
                _broadcast(batch['tokens'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['num_seqs'])
        
        _broadcast(batch['position_ids'])

    else:
        # dtype: long, long, float, bool, long
        tokens = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                             device=torch.cuda.current_device())
        labels = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                             device=torch.cuda.current_device())
        loss_mask = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.float32,
                                device=torch.cuda.current_device())
        
        attention_mask = None
        if args.create_attention_mask_in_dataloader:
            mbs = args.micro_batch_size if args.reset_attention_mask else 1
            attention_mask = torch.empty((mbs, 1, args.seq_length, args.seq_length), dtype=torch.bool,
                                        device=torch.cuda.current_device())
        position_ids = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                                   device=torch.cuda.current_device())

        num_seqs = None
        if per_seq_average:
            num_seqs = torch.empty((args.micro_batch_size,), dtype=torch.int64,
                                    device=torch.cuda.current_device()) 
            
        if args.pipeline_model_parallel_size == 1:
            _broadcast(tokens)
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(num_seqs)

        elif mpu.is_pipeline_first_stage():
            labels = None
            loss_mask = None
            num_seqs = None

            _broadcast(tokens)
            _broadcast(attention_mask)

        elif mpu.is_pipeline_last_stage():
            # Multi-Token Prediction (MTP) layers need tokens and position_ids to calculate embedding.
            # Currently the Multi-Token Prediction (MTP) layers is fixed on the last stage, so we need
            # to broadcast tokens and position_ids to all of the tensor parallel ranks on the last stage.
            if getattr(args, 'mtp_num_layers', None) is not None:
                _broadcast(tokens)
            else:
                tokens = None
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(num_seqs)

        _broadcast(position_ids)
        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'num_seqs': num_seqs
        }

    return batch
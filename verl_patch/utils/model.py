# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Utilities to create common models from huggingface
"""

import os
import re
import warnings
from typing import Dict, Optional, Type

import numpy as np
import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    MistralForSequenceClassification,
    PretrainedConfig,
    PreTrainedModel,
)
from megatron.training.checkpointing import get_checkpoint_tracker_filename, get_checkpoint_name, _get_checkpoint_format, isfile, read_metadata
from megatron.training.checkpointing import find_checkpoint_rank_0, CheckpointType
from megatron.core import dist_checkpointing
from megatron.training.utils import print_rank_0, unwrap_model
from megatron.core import parallel_state as mpu
from megatron.core.dist_checkpointing.serialization import get_default_load_sharded_strategy
from megatron.core.dist_checkpointing.serialization import StrictHandling

try:
    from modelopt.torch.opt.plugins import (
        save_modelopt_state,
        save_sharded_modelopt_state,
        restore_modelopt_state,
        restore_sharded_modelopt_state,
    )
    has_nvidia_modelopt = True
except Exception:
    has_nvidia_modelopt = False


def load_mcore_dist_weights(ddp_model, load_dir, strict=True,
                    checkpointing_context=None, skip_load_to_model_and_opt=False, is_value_model=False):
    """Load a model checkpoint and return the iteration.
    strict (bool): whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model.
    skip_load_to_model_and_opt (bool): whether to call `load_state_dict`
        for :attr:`model` and :attr:`optimizer`. In case of running FSDP2 with mcore distributed
        checkpointing, the tensors are already loaded in-place by `_load_base_checkpoint`.
    """
    state_dict, checkpoint_name, release, ckpt_type = _load_base_checkpoint(
        load_dir, rank0=True, checkpointing_context=checkpointing_context)

    ddp_model = unwrap_model(ddp_model)
    sharded_sd_metadata = dist_checkpointing.load_content_metadata(preloaded_state_dict=state_dict)
    if has_nvidia_modelopt:
        restore_modelopt_state(ddp_model, state_dict)
    else:
        raise ValueError("ModelOpt is not installed. Please install nvidia-modelopt to load modelopt state.")
    model_sd_kwargs = dict(metadata=sharded_sd_metadata)
    sharded_state_dict = generate_state_dict(ddp_model, model_sd_kwargs)
    state_dict, checkpoint_name, release, ckpt_type = _load_base_checkpoint(
        load_dir, rank0=False, sharded_state_dict=sharded_state_dict, checkpointing_context=checkpointing_context
)
    
    # Model.
    if not skip_load_to_model_and_opt:
        if len(ddp_model) == 1:
            ddp_model[0].load_state_dict(state_dict['model'], strict=strict)
        else:
            for i in range(len(ddp_model)):
                # If there is no corresponding model in the state_dict, it will be ignored.
                # It means that this is an empty stage.
                if 'model%d' % i not in state_dict:
                    continue
                ddp_model[i].load_state_dict(state_dict['model%d' % i], strict=strict)


    # Some utilities want to load a checkpoint without distributed being initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0(f'  successfully loaded checkpoint from {load_dir} '
                 f'[ t {mpu.get_tensor_model_parallel_rank() + 1}/{mpu.get_tensor_model_parallel_world_size()}, '
                 f'p {mpu.get_pipeline_model_parallel_rank() + 1}/{mpu.get_pipeline_model_parallel_world_size()} ] '
                )


def _load_base_checkpoint(
    load_dir,
    rank0=False,
    sharded_state_dict=None,
    checkpointing_context=None,
):
    """ Load the base state_dict from the given directory

    If rank0 is true, just loads rank 0 checkpoint, ignoring arguments.
    """

    iteration, release = -1, False
    tracker_filename = 'because load directory is not defined'
    if load_dir is not None:
        tracker_filename = get_checkpoint_tracker_filename(load_dir)
        if isfile(tracker_filename):
            iteration, release = read_metadata(tracker_filename)

    ckpt_format = "torch_dist"

    if not rank0:
        dist_infix = "distributed " if ckpt_format == "torch_dist" else ""
        if release:
            print_rank_0(f' loading release {dist_infix}checkpoint from {load_dir}')
        else:
            print_rank_0(
                f' loading {dist_infix}checkpoint from {load_dir} at iteration {iteration}'
            )


    return _load_global_dist_base_checkpoint(
        load_dir, rank0, sharded_state_dict, iteration, release, checkpointing_context=checkpointing_context
    )


def _load_global_dist_base_checkpoint(
    load_dir, rank0, sharded_state_dict, iteration, release, checkpointing_context=None
):
    """ Load the base state_dict from the given directory containing the global distributed checkpoint """
    if rank0:
        checkpoint_name = find_checkpoint_rank_0(load_dir, iteration, release)
        state_dict = dist_checkpointing.load_common_state_dict(checkpoint_name)
        return state_dict, checkpoint_name, release, CheckpointType.GLOBAL

    checkpoint_name = get_checkpoint_name(load_dir, iteration, release, return_base_dir=True)
    load_strategy = get_default_load_sharded_strategy(checkpoint_name)

    state_dict = dist_checkpointing.load(sharded_state_dict, checkpoint_name, load_strategy, strict='assume_ok_unexpected')
    return state_dict, checkpoint_name, release, CheckpointType.GLOBAL


def generate_state_dict(model, model_sd_kwargs):
    state_dict = {}
    for i in range(len(model)):
        key = "model"
        if len(model) > 1:
            key = f"model{i}"

        model_sd = model[i].sharded_state_dict(**(model_sd_kwargs or {}))
        state_dict[key] = model_sd

    return state_dict
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

import os
import random
import sys
from collections import defaultdict
import numpy as np
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron import update_num_microbatches
from megatron.checkpointing import (_transpose_first_dim,
                                    find_checkpoint_rank_0,
                                    get_checkpoint_tracker_filename,
                                    get_checkpoint_version, get_rng_state,
                                    read_metadata, set_checkpoint_version)
from megatron.core import mpu, tensor_parallel
from megatron.global_vars import get_args

try:
    from megatron.model import DistributedDataParallel as LocalDDP
except:
    from megatron.core.distributed import DistributedDataParallel as LocalDDP

from megatron.model import Float16Module
from megatron.utils import print_rank_0, unwrap_model


def get_checkpoint_names(checkpoints_path, iteration, use_distributed_optimizer, release=False,
                         pipeline_parallel=None, tensor_rank=None, pipeline_rank=None):
    """Determine the directory name for this rank's checkpoint."""
    if release:
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(iteration)

    # Use both the tensor and pipeline MP rank.
    if pipeline_parallel is None:
        pipeline_parallel = (mpu.get_pipeline_model_parallel_world_size() > 1)
    if tensor_rank is None:
        tensor_rank = mpu.get_tensor_model_parallel_rank()
    if pipeline_rank is None:
        pipeline_rank = mpu.get_pipeline_model_parallel_rank()

    # Use both the tensor and pipeline MP rank. If using the distributed
    # optimizer, then the optimizer's path must additionally include the
    # data parallel rank.
    if not pipeline_parallel:
        common_path = os.path.join(checkpoints_path, directory,
                                   f'mp_rank_{tensor_rank:02d}')
    else:
        common_path = os.path.join(checkpoints_path, directory,
                                   f'mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}')

    if use_distributed_optimizer:
        if release:
            model_name = os.path.join(common_path, "model_rng.pt")
        else:
            model_name = os.path.join(common_path, "model_optim_rng.pt")
        optim_name = os.path.join(
            common_path,
            "distrib_optim.pt")
    else:
        model_name = optim_name = os.path.join(common_path, "model_optim_rng.pt")
    return model_name, optim_name


def ensure_directory_exists(filename):
    """Build filename's path if it does not already exists."""
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.system(f'mkdir -p {dirname}')


def check_checkpoint_args(checkpoint_args):
    """Ensure fixed arguments for a model are the same for the input
    arguments and the one retrieved from checkpoint."""
    args = get_args()

    def _compare(arg_name, old_arg_name=None):
        if old_arg_name is not None:
            checkpoint_value = getattr(checkpoint_args, old_arg_name)
        else:
            checkpoint_value = getattr(checkpoint_args, arg_name)
        args_value = getattr(args, arg_name)
        error_message = '{} value from checkpoint ({}) is not equal to the ' \
                        'input argument' \
                        'value ({}).'.format(arg_name,
                                             checkpoint_value,
                                             args_value)
        assert checkpoint_value == args_value, error_message

    _compare('num_layers')
    _compare('hidden_size')
    _compare('num_attention_heads')
    if args.data_parallel_random_init:
        _compare('data_parallel_random_init')
    if get_checkpoint_version() < 3.0:
        _compare('tensor_model_parallel_size',
                 old_arg_name='model_parallel_size')
    if get_checkpoint_version() >= 3.0:
        _compare('tensor_model_parallel_size')
        _compare('pipeline_model_parallel_size')


def fix_query_key_value_ordering(model, checkpoint_version):
    """Fix up query/key/value matrix ordering if checkpoint
    version is smaller than 2.0
    """
    if checkpoint_version < 2.0:
        if isinstance(model, list):
            assert len(model) == 1
            model = model[0]
        for name, param in model.named_parameters():
            tmp1 = '.query_key_value.weight'
            tmp2 = '.query_key_value.bias'
            if name.endswith((tmp1, tmp2)):
                if checkpoint_version == 0:
                    fixed_param = _transpose_first_dim(param.data, 3, True,
                                                       model)
                elif checkpoint_version == 1.0:
                    fixed_param = _transpose_first_dim(param.data, 3, False,
                                                       model)
                else:
                    print_rank_0(
                        f'Invalid checkpoint version {checkpoint_version}.')
                    sys.exit()
                param.data.copy_(fixed_param)
            if name.endswith(('.key_value.weight', '.key_value.bias')):
                if checkpoint_version == 0:
                    fixed_param = _transpose_first_dim(param.data, 2, True,
                                                       model)
                elif checkpoint_version == 1.0:
                    fixed_param = _transpose_first_dim(param.data, 2, False,
                                                       model)
                else:
                    print_rank_0(
                        f'Invalid checkpoint version {checkpoint_version}.')
                    sys.exit()
                param.data.copy_(fixed_param)
        print_rank_0(' succesfully fixed query-key-values ordering for'
                     ' checkpoint version {}'.format(checkpoint_version))


def _load_base_checkpoint(load_dir, use_distributed_optimizer, rank0=False, model=None):
    """ Load the base state_dict from the given directory

    If rank0 is true, just loads rank 0 checkpoint, ignoring arguments.
    """

    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(load_dir)

    # If no tracker file, return nothing
    if not os.path.isfile(tracker_filename):
        if not rank0:
            print_rank_0(
                'WARNING: could not find the metadata file {} '.format(
                    tracker_filename))
            print_rank_0(
                '    will not load any checkpoints and will start from '
                'random')
        return None, None, False, None

    # Otherwise, read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    iteration, release = read_metadata(tracker_filename)

    # Checkpoint.
    if rank0:
        checkpoint_names = find_checkpoint_rank_0(load_dir, iteration,
                                                  use_distributed_optimizer,
                                                  release)
    else:
        checkpoint_names = get_checkpoint_names(load_dir, iteration,
                                                use_distributed_optimizer,
                                                release)
        if release:
            print_rank_0(f' loading release checkpoint from {load_dir}')
        else:
            print_rank_0(
                f' loading checkpoint from {load_dir} at iteration {iteration}'
            )

    model_checkpoint_name, optim_checkpoint_name = checkpoint_names
    # Load the checkpoint.
    args = get_args()
    try:
        model_state_dict = torch.load(model_checkpoint_name,
                                      map_location='cpu')
        if not args.no_load_optim:
            if use_distributed_optimizer or args.moe:
                optim_state_dict = torch.load(optim_checkpoint_name, map_location='cpu')
            else:
                optim_state_dict = model_state_dict
        else:
            optim_state_dict = None
    except ModuleNotFoundError:
        # For backward compatibility.
        if not rank0:
            print_rank_0(' > deserializing using the old code structure ...')
        sys.modules['fp16.loss_scaler'] = sys.modules[
            'megatron.fp16_deprecated.loss_scaler']
        sys.modules['megatron.fp16.loss_scaler'] = sys.modules[
            'megatron.fp16_deprecated.loss_scaler']
        model_state_dict = torch.load(model_checkpoint_name,
                                      map_location='cpu')
        optim_state_dict = torch.load(optim_checkpoint_name,
                                      map_location='cpu')
        sys.modules.pop('fp16.loss_scaler', None)
        sys.modules.pop('megatron.fp16.loss_scaler', None)
    except BaseException as e:
        print_rank_0('could not load the checkpoint')
        print_rank_0(e)
        sys.exit()

    # Load MoE
    if args.moe:
        if args.expert_tensor_parallelism and \
                mpu.get_tensor_model_parallel_world_size() > 1:
            # expert with tensor parallel, save to the mp_rank dir.
            moe_checkpoint_dir = os.path.dirname(model_checkpoint_name)
        else:
            # save to the root dir.
            moe_checkpoint_dir = os.path.dirname(os.path.dirname(model_checkpoint_name))
        _load_moe_state_dict(moe_checkpoint_dir, model_state_dict['model'],
                        model_list=model, mpu=mpu)


    return model_state_dict, optim_state_dict, release, optim_checkpoint_name


def load_checkpoint(model,
                    optimizer,
                    opt_param_scheduler,
                    load_arg='load',
                    strict=True):
    """Load a model checkpoint and return the iteration.
    strict (bool): whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model.
    """
    args = get_args()
    load_dir = getattr(args, load_arg)

    model = unwrap_model(model)
    model_state_dict, optim_state_dict, release, optim_checkpoint_name = \
        _load_base_checkpoint(
            load_dir,
            use_distributed_optimizer=args.use_distributed_optimizer,
            rank0=False,
            model=model)

    if model_state_dict is None:
        return 0

    # set checkpoint version
    set_checkpoint_version(model_state_dict.get('checkpoint_version', 0))

    # Set iteration.
    if args.finetune or release:
        iteration = 0
    else:
        try:
            iteration = model_state_dict['iteration']
        except KeyError:
            try:  # Backward compatible with older checkpoints
                iteration = model_state_dict['total_iters']
            except KeyError:
                print_rank_0('A metadata file exists but unable to load '
                             'iteration from checkpoint {}')
                sys.exit()

    if 'args' in model_state_dict:
        checkpoint_args = model_state_dict['args']
        args.consumed_train_samples = getattr(checkpoint_args,
                                              'consumed_train_samples', 0)
        update_num_microbatches(consumed_samples=args.consumed_train_samples)
        args.consumed_valid_samples = getattr(checkpoint_args,
                                              'consumed_valid_samples', 0)
    else:
        print_rank_0('could not find arguments in the checkpoint ...')

    # Model.
    if len(model) == 1:
        model[0].load_state_dict(model_state_dict['model'], strict=strict)
    else:
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            model[i].load_state_dict(model_state_dict['model%d' % i],
                                     strict=strict)

    # Fix up query/key/value matrix ordering if needed
    checkpoint_version = get_checkpoint_version()
    print_rank_0(f' checkpoint version {checkpoint_version}')
    fix_query_key_value_ordering(model, checkpoint_version)

    # Optimizer.
    if not release and not args.finetune and not args.no_load_optim:
        try:
            if optimizer is not None:
                optimizer.load_state_dict(model_state_dict['optimizer'])
            if optimizer is not None and args.use_distributed_optimizer:
                optimizer.load_parameter_state(optim_checkpoint_name)
            if opt_param_scheduler is not None:
                if 'lr_scheduler' in model_state_dict:  # backward compatbility
                    opt_param_scheduler.load_state_dict(
                        model_state_dict['lr_scheduler'])
                else:
                    opt_param_scheduler.load_state_dict(
                        model_state_dict['opt_param_scheduler'])
        except KeyError:
            print_rank_0('Unable to load optimizer from checkpoint. '
                         'Specify --no-load-optim or --finetune to prevent '
                         'attempting to load the optimizer state, '
                         'exiting ...')
            sys.exit()

    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            if 'rng_state' in model_state_dict:
                # access rng_state for data parallel rank
                if args.data_parallel_random_init:

                    rng_state = model_state_dict['rng_state'][
                        mpu.get_data_parallel_rank()]
                else:
                    rng_state = model_state_dict['rng_state'][0]
                random.setstate(rng_state['random_rng_state'])
                np.random.set_state(rng_state['np_rng_state'])
                torch.set_rng_state(rng_state['torch_rng_state'])
                torch.cuda.set_rng_state(rng_state['cuda_rng_state'])
                # Check for empty states array
                if not rng_state['rng_tracker_states']:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(
                    rng_state['rng_tracker_states'])
            else:  # backward compatability
                random.setstate(model_state_dict['random_rng_state'])
                np.random.set_state(model_state_dict['np_rng_state'])
                torch.set_rng_state(model_state_dict['torch_rng_state'])
                torch.cuda.set_rng_state(model_state_dict['cuda_rng_state'])
                # Check for empty states array
                if not model_state_dict['rng_tracker_states']:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(
                    model_state_dict['rng_tracker_states'])
        except KeyError:
            print_rank_0('Unable to load rng state from checkpoint. '
                         'Specify --no-load-rng or --finetune to prevent '
                         'attempting to load the rng state, '
                         'exiting ...')
            sys.exit()

    # Some utilities want to load a checkpoint
    # without distributed being initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0(f'  successfully loaded checkpoint from {args.load} '
                 f'at iteration {iteration}')

    return iteration


def get_hf_checkpoint_dir(checkpoints_path, iteration, release=False):
    """A unified checkpoint name."""
    if release:
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(iteration)

    return os.path.join(checkpoints_path, directory)


def save_checkpoint(iteration, model, optimizer, opt_param_scheduler):
    """Save a model checkpoint."""
    args = get_args()

    # Only rank zero of the data parallel writes to the disk.
    model = unwrap_model(model)

    print_rank_0('saving checkpoint at iteration {:7d} to {}'.format(
        iteration, args.save))

    # Collect rng state across data parallel ranks.
    rng_state = get_rng_state()

    # Checkpoint file names.
    model_checkpoint_name, optim_checkpoint_name = \
        get_checkpoint_names(args.save,
                             iteration,
                             args.use_distributed_optimizer)

    # Collect args, model, RNG.
    model_state_dict = {}
    if not torch.distributed.is_initialized() \
       or mpu.get_data_parallel_rank() == 0:

        # Arguments, iteration, and model.
        model_state_dict['args'] = args
        model_state_dict['checkpoint_version'] = 3.0
        model_state_dict['iteration'] = iteration
        if args.transformer_type == 'megatron':
            if len(model) == 1:
                model_state_dict['model'] = model[
                    0].state_dict_for_save_checkpoint()
                if args.moe:
                    # Remove moe states
                    model_state_dict['model']['language_model'].pop('moe_state_dict')
            else:
                for i in range(len(model)):
                    mpu.set_virtual_pipeline_model_parallel_rank(i)
                    model_state_dict['model%d' % i] = \
                        model[i].state_dict_for_save_checkpoint()
                    if args.moe:
                        model_state_dict['model%d' % i]['language_model'].pop('moe_state_dict')
        elif args.transformer_type == 'huggingface':
            model_state_dict['model'] = model[0].state_dict()

        # RNG states.
        if not args.no_save_rng:
            model_state_dict['rng_state'] = rng_state

    # Collect optimizer state.
    # (Optimizer is saved separately from the model, due
    # to the conflicting data pattern when using the distributed optimizer.)
    optim_state_dict = {}
    if not args.no_save_optim \
       and (not torch.distributed.is_initialized()
            or mpu.get_data_parallel_rank() == 0
            or args.use_distributed_optimizer):

        # Optimizer stuff.
        if optimizer is not None:
            optim_state_dict['optimizer'] = optimizer.state_dict()
        if opt_param_scheduler is not None:
            optim_state_dict['opt_param_scheduler'] = \
                opt_param_scheduler.state_dict()

    if args.transformer_type == 'megatron':
        # Save.
        if args.use_distributed_optimizer:
            # Save model separate from optimizer.
            if model_state_dict:
                ensure_directory_exists(model_checkpoint_name)
                torch.save(model_state_dict, model_checkpoint_name)
            if optim_state_dict:
                ensure_directory_exists(optim_checkpoint_name)
                torch.save(optim_state_dict, optim_checkpoint_name)
        else:
            # Save model and optimizer together.
            state_dict = {**model_state_dict, **optim_state_dict}
            if state_dict:
                ensure_directory_exists(model_checkpoint_name)
                torch.save(state_dict, model_checkpoint_name)

    elif args.transformer_type == 'huggingface':
        checkpoint_dir = get_hf_checkpoint_dir(args.save, iteration)
        ensure_directory_exists(checkpoint_dir)
        unwrapped_model = unwrap_model(model[0],
                                       (torchDDP, LocalDDP, Float16Module))
        unwrapped_model.save_pretrained(checkpoint_dir)

    if args.moe:
        if args.expert_tensor_parallelism and \
                mpu.get_tensor_model_parallel_world_size() > 1:
            # expert with tensor parallel, save to the mp_rank dir.
            moe_checkpoint_dir = os.path.dirname(model_checkpoint_name)
        else:
            # save to the root dir.
            moe_checkpoint_dir = os.path.dirname(os.path.dirname(model_checkpoint_name))
        print_rank_0('  save moe checkpoints to {}'.format(moe_checkpoint_dir))
        ensure_directory_exists(moe_checkpoint_dir)
        _save_moe_checkpoint(moe_checkpoint_dir, model)

    # Wait so everyone is done (necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0(
        '  successfully saved checkpoint at iteration {:7d} to {}'.format(
            iteration, args.save))

    # And update the latest iteration
    if not torch.distributed.is_initialized() or torch.distributed.get_rank(
    ) == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, 'w') as f:
            f.write(str(iteration))

    # Wait so everyone is done (not necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def _get_expert_ckpt_name(checkpoints_path, layer_id, expert_id, tag, mpu):
    args = get_args()
    mp_rank = 0
    if args.expert_tensor_parallelism:
        mp_rank = mpu.get_tensor_model_parallel_rank()
    # Used to support expert saving and loading.
    ckpt_name = os.path.join(
        checkpoints_path,
        '' if tag is None else str(tag),
        f'layer_{layer_id}_expert_{expert_id}_mp_rank_{mp_rank:02d}_model_states.pt'
    )
    return ckpt_name

def _load_moe_state_dict(checkpoint_path, state_dict, model_list, mpu):
    from megatron_patch.model.mixtral.layer import MoE
    from megatron_patch import expert_parallel_state
    moe_state_dict = state_dict['language_model'].setdefault('moe_state_dict', {})

    # Loop through all the models in the list
    for model in model_list:
        # Loop through all the modules in the model
        for _, module in model.named_modules():
            # Check if the module is an MoE layer
            if isinstance(module, MoE):
                moe_layer_index = module.get_moe_layer_index()
                num_local_experts = module.num_local_experts

                # Get the rank of the current process and calculate the global expert ID
                ep_rank = torch.distributed.get_rank(group=expert_parallel_state.get_expert_parallel_group())

                # Loop through all the local experts
                for local_expert_id in range(num_local_experts):
                    # Calculate the name of the checkpoint file and load the expert state dictionary
                    global_expert_id = ep_rank * num_local_experts + local_expert_id
                    expert_ckpt_name = _get_expert_ckpt_name(checkpoint_path, moe_layer_index,
                                                             global_expert_id, None, mpu)
                    expert_state_dict = torch.load(expert_ckpt_name,
                                                   map_location=torch.device('cpu'))

                    # Update the expert state dictionary with the local expert ID
                    moe_str_prefix = '.megatron_moe.experts.megatron_experts.'
                    for key in list(expert_state_dict.keys()):
                        local_key = key.replace(f'{moe_str_prefix}{global_expert_id}',
                                                f'{moe_str_prefix}{local_expert_id}')
                        expert_state_dict[local_key] = expert_state_dict.pop(key)

                    # Update the MoE state dictionary with the expert state dictionary
                    moe_state_dict.update(expert_state_dict)

def _save_moe_checkpoint(save_dir, model_list):
    # Using layer_#_export_# to save the model's expert state_dict
    from megatron_patch.model.mixtral.layer import MoE
    from megatron_patch import expert_parallel_state
    import re
    # Loop through all the models in the list
    for model in model_list:
        # Loop through all the modules in the model
        for name, module in model.named_modules():
            # Check if the module is an MoE layer
            if isinstance(module, MoE):
                moe_layer_id = module.get_moe_layer_index()
                num_local_experts = module.num_local_experts
                ep_rank = torch.distributed.get_rank(group=expert_parallel_state.get_expert_parallel_group())

                # Extract the state dict of MoE experts
                moe_state_dict = {
                    f"{name}.{n}": p
                    for n, p in module.state_dict().items()
                    if "expert" in n and "moe.gate.wg.weight" not in n
                }

                # Loop through all the experts and update the state dict with global expert IDs
                experts_state_dict = defaultdict(dict)
                for key in list(moe_state_dict.keys()):
                    match = re.match(f".*{name}.megatron_moe.experts.megatron_experts.([0-9]+).*",
                                     key)

                    if match is None:
                        print(f"No expert found in key {key}.")
                        continue

                    local_expert_id = match.group(1)
                    global_expert_id = ep_rank * num_local_experts + int(local_expert_id)

                    expert_key = key.replace(
                        f"{name}.megatron_moe.experts.megatron_experts.{local_expert_id}",
                        f"{name}.megatron_moe.experts.megatron_experts.{global_expert_id}")

                    experts_state_dict[str(global_expert_id)][expert_key] = moe_state_dict.pop(key)

                # Save the expert state dictionaries
                for global_expert_id, expert_state_dict in experts_state_dict.items():
                    save_path = _get_expert_ckpt_name(save_dir, moe_layer_id, global_expert_id,
                                                      None, mpu)
                    torch.save(expert_state_dict, save_path)

                moe_layer_id += 1
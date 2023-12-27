# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Expert parallel groups."""
import torch

_EXPERT_PARALLEL_GROUP = None
_MPU_EXPERT_PARALLEL_WORLD_SIZE = None

def initialize_moe_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    expert_tensor_parallelism: bool = False
) -> None:
    """Initialize model data parallel groups.
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    if (
        world_size
        % (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size)
        != 0
    ):
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
            f"({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size}) "
            f"x context_parallel_size ({context_parallel_size})"
        )

    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    rank = torch.distributed.get_rank()

    global _EXPERT_PARALLEL_GROUP
    assert _EXPERT_PARALLEL_GROUP is None, \
        'expert parallel group is already initialized'
    # Currently, data parallelism is not supported for experts.
    if expert_tensor_parallelism:
        # ETP + EP
        for i in range(pipeline_model_parallel_size):
            start_rank = i * num_pipeline_model_parallel_groups
            end_rank = (i + 1) * num_pipeline_model_parallel_groups
            for j in range(tensor_model_parallel_size):
                ranks = range(start_rank + j, end_rank,
                              tensor_model_parallel_size)
                group = torch.distributed.new_group(ranks)
                if rank in ranks:
                    _EXPERT_PARALLEL_GROUP = group
    else:
        # Pure EP
        for i in range(pipeline_model_parallel_size):
            start_rank = i * num_pipeline_model_parallel_groups
            end_rank = (i + 1) * num_pipeline_model_parallel_groups
            ranks = range(start_rank, end_rank)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _EXPERT_PARALLEL_GROUP = group


def get_expert_parallel_group():
    """Get the expert parallel group the caller rank belongs to."""
    assert _EXPERT_PARALLEL_GROUP is not None, \
        'expert parallel group is not initialized'
    return _EXPERT_PARALLEL_GROUP


def set_expert_parallel_world_size(world_size):
    """Set the expert parallel size"""
    global _MPU_EXPERT_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_PARALLEL_WORLD_SIZE = world_size


def get_expert_parallel_world_size():
    """Return world size for the expert parallel group."""
    global _MPU_EXPERT_PARALLEL_WORLD_SIZE
    if _MPU_EXPERT_PARALLEL_WORLD_SIZE is not None:
        return _MPU_EXPERT_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_expert_parallel_group())
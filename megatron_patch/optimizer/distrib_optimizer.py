# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from megatron.core import mpu, tensor_parallel


_EXPERT_PARALLEL_ATTRIBUTE_DEFAULTS = {
    'router': None,
    'expert_id': None
}

def copy_expert_parallel_attributes(destination_tensor, source_tensor):
    def maybe_copy(attribute):
        if hasattr(source_tensor, attribute):
            setattr(destination_tensor, attribute, getattr(source_tensor, attribute))
    for attribute in _EXPERT_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_copy(attribute)


"""Megatron distributed optimizer."""
@classmethod
def build_model_and_main_param_groups(cls, gbuf_ranges, param_gbuf_map, opt_group_ranges):
    """
    Create main parameter groups needed for the optimizer step.

    These groups encompass both: 1) groups used by this class, for
    reducing/gather, and 2) groups used by the inner optimizer for the
    parameter update. Given that the conceptual grad buffer partitioning
    (created in earlier method) doesn't respect parameter boundaries,
    the optimizer operates on shards of the model parameters, rather than
    the full parameters.
    """

    # Parameter groups:
    #   model_float16_groups: original float16 parameters
    #   model_fp32_groups: original fp32 parameters
    #   shard_float16_groups: shards of original float16 parameters
    #   shard_fp32_groups: shards of original fp32 parameters
    #   shard_fp32_from_float16_groups: fp32 copy of float16 parameters
    model_float16_groups = []
    model_fp32_groups = []
    shard_float16_groups = []
    shard_fp32_groups = []
    shard_fp32_from_float16_groups = []

    # Allocate (or slice) each group's param shard.
    for group_index, group_range in enumerate(opt_group_ranges):

        # Params of this group.
        model_float16_params_this_group = []
        model_fp32_params_this_group = []
        shard_float16_params_this_group = []
        shard_fp32_params_this_group = []
        shard_fp32_from_float16_params_this_group = []
        model_float16_groups.append(model_float16_params_this_group)
        model_fp32_groups.append(model_fp32_params_this_group)
        shard_float16_groups.append(shard_float16_params_this_group)
        shard_fp32_groups.append(shard_fp32_params_this_group)
        shard_fp32_from_float16_groups.append(shard_fp32_from_float16_params_this_group)

        for model_param in group_range["params"]:

            assert model_param.requires_grad

            gbuf_index, dtype, bucket_index = param_gbuf_map[model_param]
            gbuf_range = gbuf_ranges[gbuf_index][dtype][bucket_index]
            param_range = gbuf_range["param_map"][model_param]["param"]

            # fp16, bf16 params.
            if model_param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:

                # Clone model -> main.
                shard_model_param = model_param.detach().view(-1)[
                                    param_range.start: param_range.end
                                    ]
                shard_main_param = shard_model_param.clone().float()
                tensor_parallel.copy_tensor_model_parallel_attributes(
                    shard_model_param, model_param
                )
                tensor_parallel.copy_tensor_model_parallel_attributes(
                    shard_main_param, model_param
                )

                copy_expert_parallel_attributes(
                    shard_main_param, model_param
                )
                if hasattr(model_param, 'shared'):
                    shard_model_param.shared = model_param.shared
                    shard_main_param.shared = model_param.shared
                if hasattr(model_param, 'router'):
                    shard_model_param.router = model_param.router
                    shard_main_param.router = model_param.router

                # Add to group.
                model_float16_params_this_group.append(model_param)
                shard_float16_params_this_group.append(shard_model_param)
                shard_fp32_from_float16_params_this_group.append(shard_main_param)

            # fp32 params.
            elif model_param.type() == 'torch.cuda.FloatTensor':
                shard_model_param = model_param.view(-1)[param_range.start: param_range.end]
                model_fp32_params_this_group.append(model_param)
                shard_fp32_params_this_group.append(shard_model_param)
                tensor_parallel.copy_tensor_model_parallel_attributes(
                    shard_model_param, model_param
                )
                if hasattr(model_param, 'shared'):
                    shard_model_param.shared = model_param.shared

            else:
                raise TypeError(
                    'Wrapped parameters must be one of '
                    'torch.cuda.FloatTensor,  '
                    'torch.cuda.HalfTensor, or '
                    'torch.cuda.BFloat16Tensor. '
                    'Received {}'.format(model_param.type())
                )

        # Update optimizer's params.
        group_range["orig_group"]["params"] = [
            *shard_fp32_params_this_group,
            *shard_fp32_from_float16_params_this_group,
        ]

    return (
        model_float16_groups,
        model_fp32_groups,
        shard_float16_groups,
        shard_fp32_groups,
        shard_fp32_from_float16_groups,
    )
 
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Megatron optimizer."""
from megatron.core import  tensor_parallel
from megatron.optimizer.optimizer import Float16OptimizerWithFloat16Params

def new_init(
        self,
        optimizer,
        clip_grad,
        log_num_zeros_in_grad,
        check_for_nan_in_grad,
        params_have_main_grad,
        fp16,
        bf16,
        params_dtype,
        grad_scaler,
    ):
    origin_init = super(Float16OptimizerWithFloat16Params, self).__init__
    origin_init(
             optimizer=optimizer,
             clip_grad=clip_grad,
             log_num_zeros_in_grad=log_num_zeros_in_grad,
             check_for_nan_in_grad=check_for_nan_in_grad,
             params_have_main_grad=params_have_main_grad,
             fp16=fp16,
             bf16=bf16,
             params_dtype=params_dtype,
             grad_scaler=grad_scaler)
    # ======================
    # main parameter stuff
    # ======================

    # Three groups of parameters:
    #   float16_groups: original float16 parameters
    #   fp32_from_float16_groups: fp32 copy of float16 parameters
    #   fp32_from_fp32_groups: original fp32 parameters
    self.float16_groups = []
    self.fp32_from_float16_groups = []
    self.fp32_from_fp32_groups = []

    # For all the groups in the original optimizer:
    for param_group in self.optimizer.param_groups:
        float16_params_this_group = []
        fp32_params_this_group = []
        fp32_from_float16_params_this_group = []
        # For all the parameters in this group:
        for i, param in enumerate(param_group['params']):
            if param.requires_grad:
                # float16 params:
                if param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:
                    float16_params_this_group.append(param)
                    is_expert_parallel = not getattr(param, 'allreduce', True)
                    # Create a copy
                    main_param = param.detach().clone().float()
                    # Copy tensor model parallel attributes.
                    tensor_parallel.copy_tensor_model_parallel_attributes(main_param, param)
                    if hasattr(param, 'shared'):
                        main_param.shared = param.shared
                    # Replace the optimizer params with the new fp32 copy.
                    param_group['params'][i] = main_param
                    if is_expert_parallel:
                        setattr(main_param, 'expert_id', param.expert_id)
                        setattr(main_param, 'allreduce', True)
                    if getattr(param, 'router', False):
                        setattr(main_param, 'router', param.router)

                    fp32_from_float16_params_this_group.append(main_param)
                    # Reset existing state dict key to the new main param.
                    if param in self.optimizer.state:
                        self.optimizer.state[main_param] = self.optimizer.state.pop(param)
                # fp32 params.
                elif param.type() == 'torch.cuda.FloatTensor':
                    fp32_params_this_group.append(param)
                    param_group['params'][i] = param

                else:
                    raise TypeError(
                        'Wrapped parameters must be one of '
                        'torch.cuda.FloatTensor,  '
                        'torch.cuda.HalfTensor, or '
                        'torch.cuda.BFloat16Tensor. '
                        'Received {}'.format(param.type())
                    )

        self.float16_groups.append(float16_params_this_group)
        self.fp32_from_float16_groups.append(fp32_from_float16_params_this_group)
        self.fp32_from_fp32_groups.append(fp32_params_this_group)
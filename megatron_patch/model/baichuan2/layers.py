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

import warnings
from typing import Optional

import torch
import os
from torch.cuda.amp import custom_bwd, custom_fwd
from megatron.core.parallel_state import (
    get_global_memory_buffer,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

_grad_accum_fusion_available = True
try:
    import fused_weight_gradient_mlp_cuda
except ImportError:
    _grad_accum_fusion_available = False

_grad_accum_fusion_available = True

def linear_with_grad_accumulation_and_async_allreduce(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool,
    async_grad_allreduce: bool,
    sequence_parallel: bool,
) -> torch.Tensor:
    """Linear layer execution with asynchronous communication and
    gradient accumulation fusion in backprop.

    This has the option to accumulate the result of backprop
    calculation into an existing gradient buffer, preventing the need
    to do an additional addition kernel after the gradient
    calculation.

    Additionally, the tensor parallel all reduce of the input
    gradients can be done asynchronously with the calculation of
    the weight gradients.

    In the case of sequence parallelism, the reduce scatter of the
    input gradients is done asynchronously with the calcluation of the
    weight gradients.

    Use of this module requires that the environment variable
    CUDA_DEVICE_MAX_CONNECTIONS=1. There are a few collective
    operations, noted in the code, that should be scheduled before
    compute kernels to overlap the communication with the computation,
    which is necessary for a speedup but not for correctness so that
    ordering isn't imposed by the scheduler. Setting
    CUDA_DEVICE_MAX_CONNECTIONS=1 forces the kernels to be scheduled
    in the order they are called.

    Arguments:

    input (torch.Tensor required): input like torch.nn.functional.linear

    weight (torch.Tensor required): weight like torch.nn.functional.linear

    bias (torch.Tensor optional): bias like torch.nn.functional.linear

    gradient_accumulation_fusion (bool required): Perform the gradient
        accumulation fusion, requires the custom CUDA extension
        fused_weight_gradient_mlp_cuda module. To use
        gradient_accumulation_fusion you must install APEX with
        --cpp_ext and --cuda_ext. For example: "pip install
        --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\"
        " Note that the extension requires CUDA>=11. Otherwise, you
        must turn off gradient accumulation fusion."

    async_grad_allreduce (bool required): Do the allreduce of input
        gradients asyncronously with the computation of weight
        gradients. If sequence_parallel is True, this must be
        False, as no all reduce is performed.

    sequence_parallel (bool required): Indicates that sequence
        parallelism is used and thus in the forward pass the input is
        all gathered, and the backward pass the input gradients are
        reduce scattered.
    """
    args = [
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        async_grad_allreduce,
        sequence_parallel,
    ]

    if not linear_with_grad_accumulation_and_async_allreduce.warned:
        if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":
            if sequence_parallel:
                warnings.warn(
                    "When using sequence parallelism it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup"
                )
                linear_with_grad_accumulation_and_async_allreduce.warned = True

            if async_grad_allreduce:
                warnings.warn(
                    "When using async grad allreduce it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup"
                )
                linear_with_grad_accumulation_and_async_allreduce.warned = True

    return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)

linear_with_grad_accumulation_and_async_allreduce.warned = False

class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """See linear_with_grad_accumulation_and_async_allreduce"""

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        async_grad_allreduce,
        sequence_parallel,
    ):
        #ctx.save_for_backward(input, weight)
        norm = torch.sqrt(torch.sum(weight.pow(2), axis=1)).reshape(weight.size(0), 1).expand(weight.size(0), weight.size(1))
        ctx.save_for_backward(input, weight, norm)
        weight = torch.nn.functional.normalize(weight)
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.sequence_parallel = sequence_parallel

        if sequence_parallel:
            world_size = get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            torch.distributed._all_gather_base(
                all_gather_buffer, input, group=get_tensor_model_parallel_group()
            )
            total_input = all_gather_buffer
        else:
            total_input = input

        output = torch.matmul(total_input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        #input, weight = ctx.saved_tensors
        input, weight, norm = ctx.saved_tensors
        use_bias = ctx.use_bias

        if ctx.sequence_parallel:
            world_size = get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            handle = torch.distributed._all_gather_base(
                all_gather_buffer, input, group=get_tensor_model_parallel_group(), async_op=True
            )

            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # gather is scheduled before the input gradient computation
            total_input = all_gather_buffer
        else:
            total_input = input
        grad_input = grad_output.matmul(weight)

        if ctx.sequence_parallel:
            handle.wait()

        # Doing gather + slicing during the NeMo forward pass can make this tensor
        # not be contiguous. PyTorch only checks if the tensor is contiguous, and only
        # clones it if it's not contiguous:
        # https://github.com/pytorch/pytorch/blob/c47cf9bc7f9e02f649ab4ed53fe4d35732c92ab6/torch/_refs/__init__.py#L2761
        grad_output = grad_output.contiguous()
        # Convert the tensor shapes to 2D for execution compatibility
        grad_output = grad_output.view(
            grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
        )
        total_input = total_input.view(
            total_input.shape[0] * total_input.shape[1], total_input.shape[2]
        )

        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(
                grad_input, group=get_tensor_model_parallel_group(), async_op=True
            )
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # all-reduce is scheduled before the weight gradient computation

        if ctx.sequence_parallel:
            assert not ctx.async_grad_allreduce
            dim_size = list(input.size())
            sub_grad_input = torch.empty(
                dim_size, dtype=input.dtype, device=torch.cuda.current_device(), requires_grad=False
            )
            # reduce_scatter
            handle = torch.distributed._reduce_scatter_base(
                sub_grad_input, grad_input, group=get_tensor_model_parallel_group(), async_op=True
            )
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # reduce scatter is scheduled before the weight gradient computation

        grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None
        rs = torch.sum(weight, axis=1).reshape(weight.size(0), 1).expand(-1, weight.size(1))
        scale = (-weight * rs + norm.pow(2)) / norm.pow(3)
        if ctx.sequence_parallel:
            handle.wait()
            #return sub_grad_input, grad_weight, grad_bias, None, None, None
            return sub_grad_input, grad_weight * scale, grad_bias, None, None, None


        if ctx.async_grad_allreduce:
            handle.wait()

        #return grad_input, grad_weight, grad_bias, None, None, None
        return grad_input, grad_weight * scale, grad_bias, None, None, None
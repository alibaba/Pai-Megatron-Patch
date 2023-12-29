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
from megatron_patch.expert_parallel_state import get_expert_parallel_world_size

class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        """
        The forward pass for the all-to-all communication operation.
        
        Args:
            ctx: The context object that can be used to stash information
                 for backward computation.
            group: The process group to which the collective operation is applied.
            input: The input tensor to be split across devices.
            output_split_sizes: A tuple or list specifying the sizes of the output
                                chunks for each device after the all-to-all operation.
            input_split_sizes: A tuple or list specifying the sizes of the input
                               chunks for each device before the all-to-all operation.
        
        Returns:
            The output tensor with chunks of data received from all devices.
        """
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = get_expert_parallel_world_size()
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input

        input = input.contiguous()
        if output_split_sizes is None:
            # Equal split (all2all)
            output = torch.empty_like(input)
        else:
            # Unequal split (all2all-v)
            output = input.new_empty(
                size=[sum(output_split_sizes)] + list(input.size()[1:]),
                dtype=input.dtype,
                device=torch.cuda.current_device())
        torch.distributed.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group)
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        """
        The backward pass for the all-to-all communication operation.
        
        Args:
            ctx: The context object with saved information from the forward pass.
            grad_output: The gradient tensor with respect to the output of the
                         forward pass.
        
        Returns:
            Tuple containing None for non-tensor inputs and the gradient with
            respect to the input tensor of the forward pass.
        """
        return (None, _AllToAll.apply(
            ctx.group, * grad_output,
            ctx.input_split_sizes,
            ctx.output_split_sizes), None, None)

def all_to_all(group, input_, output_split_sizes_=None, input_split_sizes_=None):
    return _AllToAll.apply(group, input_, output_split_sizes_, input_split_sizes_)

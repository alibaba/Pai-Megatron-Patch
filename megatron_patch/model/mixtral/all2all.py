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
    def forward(ctx,
                group,
                input,
                output_split_sizes,
                input_split_sizes):
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
        return (None, _AllToAll.apply(
            ctx.group, * grad_output,
            ctx.input_split_sizes,
            ctx.output_split_sizes), None, None)

def all_to_all(group, input_, output_split_sizes_=None, input_split_sizes_=None):
    return _AllToAll.apply(group, input_, output_split_sizes_, input_split_sizes_)

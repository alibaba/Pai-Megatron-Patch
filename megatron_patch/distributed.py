# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch
import math
from contextlib import contextmanager
from typing import Dict

from megatron.core import mpu
from megatron.model.distributed import DistributedDataParallelBase
from megatron.model.distributed import GradBuffer


def is_moe_param(param: torch.Tensor) -> bool:
    if hasattr(param, "allreduce") and not param.allreduce:
        return True
    return False

class DistributedDataParallel(DistributedDataParallelBase):
    """
    DDP wrapper which stores grads in contiguous buffers. Also has option of
    overlapping communication with backprop computation by breaking up full model's
    gradients into smaller buckets and running all-reduce / reduce-scatter
    on each bucket asynchronously.
    This class:
        - has the potential to reduce memory fragmentation.
        - provides the option to do the gradient accumulation
          in a type other than the params type (e.g., fp32).

    Arguments:
        module: input model.
        data_parallel_group: data-parallel group.
        accumulate_allreduce_grads_in_fp32: if true do the gradient accumulation
            and communication in float32.
        overlap_grad_reduce: if true, overlap communication with backprop
            computation by breaking up grads into buckets. If false, single
            synchronous communication call is used instead.
        use_distributed_optimizer: if true, issue reduce-scatter communication
            calls as part of distributed optimizer. If false, issue all-reducde
            communication calls.

    """

    def __init__(
        self,
        module: torch.nn.Module,
        data_parallel_group: torch.distributed.ProcessGroup,
        accumulate_allreduce_grads_in_fp32: bool,
        overlap_grad_reduce: bool,
        use_distributed_optimizer: bool,
        bucket_size: int = 40000000,
    ):
        super(DistributedDataParallel, self).__init__(module)

        # Set bucket_size to infinity if overlap_grad_reduce is False.
        self.overlap_grad_reduce = overlap_grad_reduce
        self.use_distributed_optimizer = use_distributed_optimizer

        if not self.overlap_grad_reduce:
            bucket_size = None
        self.bucket_size = bucket_size

        self.module = module
        self.grad_buffers = {}
        self.expert_params = []
        self.expert_grads = []
        self.grad_buffer_param_index_map = {}
        self.param_to_grad_buffer = {}

        # Group parameters by their gradient type.
        grad_dtype_to_params = {}
        grad_dtype_to_numel = {}
        param_to_name = {}
        for name, param in self.module.named_parameters():
            if param.requires_grad and getattr(param, 'allreduce', True):
                param.grad_added_to_main_grad = False
                param_to_name[param] = name
                dtype = torch.float if accumulate_allreduce_grads_in_fp32 else param.dtype

                params = grad_dtype_to_params.get(dtype, [])
                params.append(param)
                grad_dtype_to_params[dtype] = params

                # Calculate number of elements per dtype.
                grad_dtype_to_numel[dtype] = (
                    grad_dtype_to_numel.get(dtype, 0) + param.data.nelement()
                )

        # Allocate the grad buffers and map the grads.
        # The grad buffer under the hood creates buckets as appropriate, depending on
        # whether overlap_grad_reduce is True or not.
        data_parallel_world_size = torch.distributed.get_world_size(group=data_parallel_group)
        for dtype, params in grad_dtype_to_params.items():
            # Pad so size is divisible by the data parallel size.
            numel = grad_dtype_to_numel[dtype]
            numel_padded = (
                int(math.ceil(numel / data_parallel_world_size)) * data_parallel_world_size
            )

            self.grad_buffers[dtype] = GradBuffer(
                numel,
                numel_padded,
                dtype,
                params,
                data_parallel_group,
                bucket_size,
                param_to_name,
                self.overlap_grad_reduce,
                self.use_distributed_optimizer,
            )

            # Parameters are laid out in the corresponding grad_buffer in reverse
            # order, so count indices from the back.
            index = grad_dtype_to_numel[dtype]
            for param in params:
                self.param_to_grad_buffer[param] = self.grad_buffers[dtype]
                if dtype not in self.grad_buffer_param_index_map:
                    self.grad_buffer_param_index_map[dtype] = {}

                index -= param.data.nelement()
                # Store the indices / bucket of each param.
                self.grad_buffer_param_index_map[dtype][param] = (
                    index,
                    index + param.data.nelement(),
                    self.grad_buffers[dtype].param_to_bucket_index[param],
                )

        # Allocate discreate buffer for MoE params' grads
        for param in self.module.parameters():
            if param.requires_grad and not getattr(param, 'allreduce', True):
                dtype = torch.float if accumulate_allreduce_grads_in_fp32 else param.dtype
                param.main_grad = \
                    torch.zeros(param.data.shape,
                                dtype=dtype,
                                device=torch.cuda.current_device(),
                                requires_grad=False)
                self.expert_grads.append(param.main_grad)
                self.expert_params.append(param)


        # Register backward hook.
        # Accumulation function for the gradients need to be stored so they
        # don't go out of scope.
        self.grad_accs = []
        for param in self.module.parameters():
            if param.requires_grad:
                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param)
                # Get the gradient accumulator function.
                grad_acc = param_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_param_hook(param, self.param_to_grad_buffer))
                self.grad_accs.append(grad_acc)

    def _make_param_hook(
        self, param: torch.nn.Parameter, param_to_grad_buffer: Dict[torch.nn.Parameter, GradBuffer]
    ):
        """Create the all-reduce / reduce-scatter hook for backprop."""

        def param_hook(*unused):
            if param.requires_grad:
                if self.overlap_grad_reduce:
                    assert (
                        param.grad is not None
                    ), 'param.grad being None is not safe when overlap_grad_reduce is True'
                if param.grad is not None and not param.grad_added_to_main_grad:
                    param.main_grad.add_(param.grad.data)
                param.grad = None
                if self.overlap_grad_reduce:
                    param_to_grad_buffer[param].mark_grad_as_done(param)

        return param_hook

    @contextmanager
    def no_sync(self):
        """Context manager that turns off gradient synchronization."""
        for grad_buffer in self.grad_buffers.values():
            grad_buffer.is_last_microbatch = False
        try:
            yield
        finally:
            for grad_buffer in self.grad_buffers.values():
                grad_buffer.is_last_microbatch = True

    def grad_sync(self, *unused):
        """Method to dispatch grad sync operations."""
        for grad_buffer in self.grad_buffers.values():
            grad_buffer.grad_sync()

    def zero_grad_buffer(self):
        """Set the grad buffer data to zero. Needs to be called at the
        begining of each iteration."""
        for param in self.module.parameters():
            if param.requires_grad:
                param.grad_added_to_main_grad = False
        for grad_buffer in self.grad_buffers.values():
            grad_buffer.reset()
        for expert_grad in self.expert_grads:
            expert_grad.zero_()

    def broadcast_params(self):
        """Sync params across all DP ranks."""
        for param in self.module.parameters():
            torch.distributed.broadcast(
                param.data,
                src=mpu.get_data_parallel_src_rank(),
                group=mpu.get_data_parallel_group(),
            )

    def sync_gradients(self):
        """
        Reduce gradients across data-parallel ranks.
        When overlap_grad_reduce is set to True, waits for asynchronous
        communication calls to complete.
        When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for grad_buffer in self.grad_buffers.values():
            grad_buffer.done()

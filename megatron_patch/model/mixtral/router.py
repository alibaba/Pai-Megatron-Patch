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

from collections.abc import Callable
from typing import Dict, TYPE_CHECKING, Any, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from megatron import get_args
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region
)

from megatron_patch import expert_parallel_state
from .all2all import all_to_all

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

uniform_map: Dict[torch.device, Callable] = {}
gumbel_map: Dict[torch.device, Callable] = {}
exp_selection_uniform_map: Dict[torch.device, Callable] = {}
 
try:
    from tutel import moe as tutel_moe
    TUTEL_INSTALLED = True
except:
    TUTEL_INSTALLED = False
    pass

# einsum rewrites are on par or more performant
# switch can be bubbled up in future
USE_EINSUM = True
# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.
def einsum(rule, a, b):
    if USE_EINSUM:
        return torch.einsum(rule, a, b)
    elif rule == 's,se->se':
        return a.reshape(a.shape[0], -1) * b
    elif rule == 'se,sc->sec':
        return a.unsqueeze(2) * b.unsqueeze(1)
    elif rule == 'se,sec->sec':
        return a.unsqueeze(2) * b
    elif rule == 'se,se->s':
        return torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).reshape(-1)
    elif rule == 'sec,sm->ecm':
        s = a.shape[0]
        e = a.shape[1]
        c = a.shape[2]
        m = b.shape[1]
        return torch.matmul(a.reshape(s, -1).t(), b).reshape(e, c, m)
    elif rule == 'sec,ecm->sm':
        return torch.matmul(a.reshape(a.shape[0], -1), b.reshape(-1, b.shape[-1]))
    elif rule == 'ks,ksm->sm':
        k = b.shape[0]
        s = b.shape[1]
        m = b.shape[2]
        # [k, s] -> [s, k] -> [s, 1, k]
        a = a.t().unsqueeze(1)
        # [k,s,m] -> [k, sm] -> [sm, k] -> [s, m, k]
        b = b.reshape(k, -1).t().reshape(s, m, k)
        # bmm([s, 1, k], [s, m, k]^t) -> [s, m, 1]
        return torch.bmm(a, b.transpose(1, 2)).squeeze(2)
    else:
        return torch.einsum(rule, a, b)

def scatter_tokens_to_tensor_parallel_region(input_):
    # E, C, M -> C, E, M
    input_ = input_.transpose(0, 1).contiguous()
    input_ = scatter_to_sequence_parallel_region(input_)
    # C, E, M -> E, C, M
    input_ = input_.transpose(0, 1).contiguous()
    return input_

def gather_tokens_from_tensor_parallel_region(input_):
    input_ = input_.transpose(0, 1).contiguous()
    input_ = gather_from_sequence_parallel_region(input_, tensor_parallel_output_grad=False)
    input_ = input_.transpose(0, 1).contiguous()
    return input_

class AuxLossBackwardHook(torch.autograd.Function):
    main_loss_backward_scale = 1
    
    @staticmethod
    def forward(ctx, output, aux_loss):
        # Preserve the aux_loss by storing it in the context to avoid garbage collection.
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Scale the auxiliary loss like the main loss.
        args = get_args()
        aux_loss, = ctx.saved_tensors
        
        aux_loss_backward_scale = AuxLossBackwardHook.main_loss_backward_scale
        if args.sequence_parallel and not args.expert_tensor_parallelism:
            # When using the sequence partitioned activation directly as the input to the Gate,
            # we need normalize the loss with regard to the number of input segements
            # (tensor_model_parallel_size). See our MR for the details.
            aux_loss_backward_scale /= args.tensor_model_parallel_size

        scaled_aux_loss_grad = torch.ones_like(aux_loss) * aux_loss_backward_scale
        return grad_output, scaled_aux_loss_grad
    
    @staticmethod
    def set_main_loss_backward_scale(scale):
        # No matter how the Main loss scales, the Aux loss needs to be scaled in the same way to
        # ensure that the gradients produced by both are scaled equally.
        AuxLossBackwardHook.main_loss_backward_scale = scale

def multiplicative_jitter(x, device: torch.device, epsilon=1e-2):
    """
    Modified from switch transformer paper. mesh transformers
    Multiply values by a random number between 1-epsilon and 1+epsilon.
    Makes models more resilient to rounding errors introduced by bfloat16.
    This seems particularly important for logits.
    Args:
        x: a torch.tensor
        device: torch.device
        epsilon: a floating point value
    Returns:
        a jittered x.
    """
    if epsilon == 0:
        return x
    uniform = uniform_map.get(device)
    if uniform is None:
        uniform = torch.distributions.uniform.Uniform(
            low=torch.tensor(1.0 - epsilon,
                             device=device),
            high=torch.tensor(1.0 + epsilon,
                              device=device)).rsample  # type: ignore
        uniform_map[device] = uniform
    return x * uniform(x.shape)
 
 
def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:                                                                                                                                                
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)


@torch.jit.script
def _capacity(gates: Tensor, capacity_factor: Tensor, min_capacity: Tensor) -> Tensor:
    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # to(torch.int64) works around a bug in torch.onnx.export:
    # it should cast k to int64 when converting torch.topk but it doesn't.
    capacity = torch.ceil((num_tokens / num_experts) * capacity_factor).to(torch.int64)
    if capacity < min_capacity:
        capacity = min_capacity.to(torch.int64)
    return capacity
 
 
@torch.jit.script
def _top_idx(source, k):
    return torch.topk(source, k=k, dim=0)[1]
 
 
@torch.jit.script
def _one_hot_to_float(x, num_classes):
    return F.one_hot(x, num_classes=num_classes).float()


# Implemented by refer to this paper: https://arxiv.org/pdf/2202.09368.pdf
def expert_choice(logits: Tensor,
                  capacity_factor: float,
                  min_capacity: float,
                  used_token: Tensor = None,
                  noisy_gate_policy: Optional[str] = None,
                  drop_tokens: bool = True,
                  use_rts: bool = True,
                  use_tutel: bool = False) -> Tuple[Tensor,
                                                    Tensor,
                                                    Tensor]:

    """ Implements Expert Choice Routing """
    # min_capacity, used_token noisy_gate_policy, use_rts and use_tutel are not used in this router
    # keep them as parameters for compatibility

    scores = F.softmax(logits, dim=1)
    # from [T, E] to [E, T]
    scores = torch.transpose(scores, 0, 1).contiguous()
    k = int(scores.shape[1] * capacity_factor / scores.shape[0])
    gatings, indices = torch.topk(scores, k=k, dim=1)

    return 0, gatings, indices


def top1gating(logits: Tensor, capacity_factor: float, min_capacity: int, used_token: Tensor = None, 
               noisy_gate_policy: Optional[str] = None, drop_tokens: bool = True, use_rts: bool = True,
               use_tutel: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top1Gating on logits."""
    
    if noisy_gate_policy == 'RSample':
        logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    
    gates = F.softmax(logits, dim=1)
 
    capacity = _capacity(gates,
                         torch.tensor(capacity_factor),
                         torch.tensor(min_capacity))
 
    # Create a mask for 1st's expert per token
    # noisy gating
    indices1_s = torch.argmax(
        logits_w_noise if noisy_gate_policy == 'RSample' else gates,
        dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)
 
    # mask only used tokens
    if used_token is not None:
        mask1 *= used_token.unsqueeze(1)
 
    # if we don't want to drop any tokens
    if not drop_tokens:
        from torch import distributed as dist
        exp_counts = mask1.sum(dim=0).detach().cpu()
        new_capacity = torch.max(exp_counts).to(logits.device)
        dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=dist.get_world_group())
        capacity = new_capacity
 
    # Compute l_aux
    me = gates.mean(dim=0)
    ce = mask1.float().mean(dim=0)
    l_aux = (me * ce).sum() * num_experts

    # Random Token Selection
    if use_rts:
        uniform = exp_selection_uniform_map.get(logits.device)
        if uniform is None:
            uniform = torch.distributions.uniform.Uniform(
                low=torch.tensor(0.0, device=logits.device),
                high=torch.tensor(1.0, device=logits.device)).rsample
            exp_selection_uniform_map[logits.device] = uniform
        mask1_rand = mask1 * uniform(mask1.shape)
    else:
        mask1_rand = mask1

    assert logits.shape[0] >= min_capacity, "No. of tokens (batch-size) should be greater than min_capacity. \
            Either set min_capacity to 0 or increase your batch size."
    top_idx = _top_idx(mask1_rand, capacity)
    new_mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx, 1)
    gatings, indices = torch.topk(gates.T.contiguous(), k=capacity, dim=1, sorted=False)
    return l_aux, gatings, indices, new_mask1


# This function has been adapted from deepspeed file:
#   https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/sharded_moe.py
def top1gating_tutel(logits: Tensor,
               capacity_factor: float,
               min_capacity: int,
               used_token: Tensor = None,
               noisy_gate_policy: Optional[str] = None,
               drop_tokens: bool = True,
               use_rts: bool = True,
               use_tutel: bool = False) -> Tuple[Tensor,
                                                 Tensor,
                                                 Tensor]:

    """Implements Top1Gating on logits."""
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)
 
    capacity = _capacity(gates,
                         torch.tensor(capacity_factor),
                         torch.tensor(min_capacity))
 
    # Create a mask for 1st's expert per token
    # noisy gating
    indices1_s = torch.argmax(
        logits_w_noise if noisy_gate_policy == 'RSample' else gates,
        dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)
 
    # mask only used tokens
    if used_token is not None:
        mask1 = torch.einsum("s,se->se", used_token, mask1)
 
    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')
 
    # if we don't want to drop any tokens
    if not drop_tokens:
        from torch import distributed as dist
        new_capacity = torch.max(exp_counts).to(logits.device)
        dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=dist.get_world_group())
        capacity = new_capacity
 
    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    # Revisit: whether to divide l_aux by micro-batches or not?
    l_aux = torch.sum(me * ce) * num_experts

    # Random Token Selection
    if use_rts:
        uniform = exp_selection_uniform_map.get(logits.device)
        if uniform is None:
            uniform = torch.distributions.uniform.Uniform(
                low=torch.tensor(0.0,
                                 device=logits.device),
                high=torch.tensor(1.0,
                                  device=logits.device)).rsample
            exp_selection_uniform_map[logits.device] = uniform

        mask1_rand = mask1 * uniform(mask1.shape)
    else:
        mask1_rand = mask1

    assert logits.shape[0] >= min_capacity, "No. of tokens (batch-size) should be greater than min_capacity. \
            Either set min_capacity to 0 or increase your batch size."
    top_idx = _top_idx(mask1_rand, capacity)

    new_mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx, 1)
    mask1 = new_mask1
    
    # Tutel doesn't support index values masked with zero
    # so we need to replace masked indices with -1
    indices_mask = mask1.sum(dim=1) * num_experts - 1
    indices1_s = torch.min(indices1_s, indices_mask)

    # Compute locations in capacity buffer

    locations1 = tutel_moe.fast_cumsum_sub_one(mask1)

    gates1_s = (gates * mask1).sum(dim=1)
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    return l_aux, capacity, num_experts, [indices1_s,], [locations1_s,], [gates1_s,]

def top2gating(logits: Tensor, capacity_factor: float, min_capacity: int, used_token: Tensor = None,
               noisy_gate_policy: Optional[str] = None, drop_tokens: bool = True, use_rts: bool = True,
               use_tutel: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)

    capacity = _capacity(gates,
                         torch.tensor(capacity_factor * 2),
                         torch.tensor(min_capacity))

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # Create a mask for 2nd's expert per token using Gumbel-max trick
    # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # Replace top-expert with min value
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float("-inf"))
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = F.one_hot(indices2_s, num_classes=num_experts)

    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.mean(me * ce) * num_experts * num_experts

    # Remove locations outside capacity from mask
    mask1 *= torch.lt(locations1, capacity)
    mask2 *= torch.lt(locations2, capacity)

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.float()
    mask2_float = mask2.float()
    gates1_s = einsum("se,se->s", gates, mask1_float)
    gates2_s = einsum("se,se->s", gates, mask2_float)
    denom_s = gates1_s + gates2_s
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s

    # Calculate combine_weights and dispatch_mask
    gates1 = einsum("s,se->se", gates1_s, mask1_float)
    gates2 = einsum("s,se->se", gates2_s, mask2_float)
    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    locations2_sc = _one_hot_to_float(locations2_s, capacity)
    combine1_sec = einsum("se,sc->sec", gates1, locations1_sc)
    combine2_sec = einsum("se,sc->sec", gates2, locations2_sc)
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask, exp_counts

# Copy from Megatron MoE branch https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/tree/moe
def sinkhorn(cost, tol=0.0001):                                                                                                        
    cost = torch.exp(cost)
    d0 = torch.ones(cost.size(0), device=cost.device, dtype=cost.dtype)
    d1 = torch.ones(cost.size(1), device=cost.device, dtype=cost.dtype)
     
    eps = 0.00000001
    error = 1e9
    d1_old = d1          
    while error > tol:   
        d0 = (1/d0.size(0))*1/(torch.sum(d1*cost,1) + eps)
        d1 = (1/d1.size(0))*1/(torch.sum(d0.unsqueeze(1)*cost,0)+eps)
        error = torch.mean(torch.abs(d1_old-d1))
        d1_old = d1      
    return d1*cost*d0.unsqueeze(1)


# Sinkhorn implementation refer to this paper:
# Unified scaling law for large language model (https://arxiv.org/pdf/2202.01169.pdf)
def sinkhornv2(logits, tol=0.01):
    f = torch.zeros(logits.size(0), device=logits.device, dtype=logits.dtype)
    g = torch.zeros(logits.size(1), device=logits.device, dtype=logits.dtype)

    # ToDo: add iteration early stop
    for _ in range(50):
        f = -torch.log(1/logits.size(1) * torch.sum(torch.exp(logits + g[None,:]), dim=1))
        g = -torch.log(1/logits.size(0) * torch.sum(torch.exp(logits + f[:, None]), dim=0))
        gates = torch.exp(logits + f[:, None] + g[None,:])
    return gates


def s_base(logits: Tensor,
           capacity_factor: float,
           min_capacity: int,
           used_token: Tensor = None,
           noisy_gate_policy: Optional[str] = None,
           drop_tokens: bool = True,
           use_rts: bool = True,
           use_tutel: bool = False) -> Tuple[Tensor,
                                             Tensor,
                                             Tensor]:
    """Implements s-base on logits."""
    # used_token, drop_token, noisy_gate_policy, drop_tokens and use_rts are not used in this router
    # keep them as paramaters for compatibility
 
    # reference: https://arxiv.org/pdf/2209.15466.pdf
    # "As in Clark (2022), we linearly combine the output of experts using a softmax matrix softmax(WX)"
    gates = F.softmax(logits, dim=1)
 
    with torch.no_grad():
        # Both sinkhorn implementations work fine
        # we choose sinkhornv2 as default here
        sinkroute = sinkhornv2(logits.detach().to(dtype=torch.float32))
        _, indices1_s = torch.max(sinkroute, dim=1)
 
    capacity = _capacity(logits,
                         torch.tensor(capacity_factor),
                         torch.tensor(min_capacity))
 
    num_experts = int(logits.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)
 
    # gating decisions                                                                                                       
    mask1_rand = mask1
    top_idx = _top_idx(mask1_rand, capacity)
    new_mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx, 1)
    mask1 = new_mask1

    if use_tutel:
        # Tutel doesn't support index values masked with zero
        # so we need to replace masked indices with -1
        indices_mask = mask1.sum(dim=1) * num_experts - 1
        indices1_s = torch.min(indices1_s, indices_mask)

    # Compute locations in capacity buffer
    if use_tutel:
        locations1 = tutel_moe.fast_cumsum_sub_one(mask1)
    else:
        locations1 = torch.cumsum(mask1, dim=0) - 1

    if use_tutel:
        sinkroute1_s = (sinkroute * mask1).sum(dim=1)
        locations1_s = torch.sum(locations1 * mask1, dim=1)
        return 0, capacity, num_experts, [indices1_s,], [locations1_s,], [sinkroute1_s,]

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.float()
    sinkroute = sinkroute * mask1_float
    gates = gates * mask1_float

    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    # reference: https://arxiv.org/pdf/2209.15466.pdf
    # "As in Clark (2022), we linearly combine the output of experts using a softmax matrix softmax(WX)"
    combine_weights = torch.einsum("se,sc->sec", gates, locations1_sc)
    dispatch_mask = torch.einsum("se,sc->sec", sinkroute, locations1_sc)
    dispatch_mask = dispatch_mask.bool()

    return 0, combine_weights, dispatch_mask


class Router(Module):
    """ Gate / Router module """
    
    def __init__(self,
                 model_dim: int,
                 num_experts: int,
                 k: int = 1,
                 capacity_factor: float = 1.0,
                 eval_capacity_factor: float = 1.0,
                 min_capacity: int = 8,
                 noisy_gate_policy: Optional[str] = None,
                 drop_tokens: bool = True,
                 use_rts: bool = True) -> None:
        super().__init__()

        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False).float()
        setattr(self.wg.weight, "router", True)
        self.k = k
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.min_capacity = min_capacity
        self.noisy_gate_policy = noisy_gate_policy
        self.drop_tokens = drop_tokens
        self.use_rts = use_rts
        args = get_args()
        self.sequence_parallel = args.sequence_parallel
        self.expert_tensor_parallelism = args.expert_tensor_parallelism
        self.tensor_model_parallel_size = args.tensor_model_parallel_size
        self.use_tutel = args.use_tutel
        self.moe_loss_coeff = args.moe_loss_coeff
        self.router_type = args.router_type

        if self.router_type == 'topk':
            if self.k == 1:
                if self.use_tutel:
                    self.gate = top1gating_tutel
                else:
                    self.gate = top1gating
            elif self.k == 2:
                self.gate = top2gating
        elif self.router_type == 'expert_choice':
            self.gate = expert_choice

    def forward(self,
                input: torch.Tensor,
                used_token: torch.Tensor = None,
                use_tutel: bool = False) -> Tuple[Tensor,
                                                  Tensor,
                                                  Tensor]:

        if self.wg.weight.dtype != torch.float32:
           self.wg = self.wg.float()
           setattr(self.wg.weight, 'router', True)
        input_fp32 = input.float()
        logits = self.wg(input_fp32)

        gate_output = self.gate(
                logits,
                self.capacity_factor if self.training else self.eval_capacity_factor,
                self.min_capacity,
                used_token,
                self.noisy_gate_policy if self.training else None,
                self.drop_tokens,
                self.use_rts,
                self.use_tutel)

        if self.router_type == 'top1':
            gate_output[0].mul_(self.moe_loss_coeff)
        
        return gate_output

class MOELayer(Base):
    """MOELayer module"""

    def __init__(self,
                 gate: Module,
                 experts: Module,
                 ep_group_name,
                 ep_size,
                 num_local_experts: int,
                 use_tutel: bool = False,
                 expert_tensor_parallelism: bool = False) -> None:
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.ep_group = None
        self.ep_size = ep_size
        self.ep_group_name = ep_group_name
        self.num_local_experts = num_local_experts
        self.num_experts = num_local_experts * ep_size
        self.ep_group = expert_parallel_state.get_expert_parallel_group()
        self.expert_tensor_parallelism = expert_tensor_parallelism

        args = get_args()
        self.router_type = args.router_type
        self.use_tutel = use_tutel and TUTEL_INSTALLED
        if self.use_tutel:
            print('Using Tutel optimizations.')
        elif use_tutel and not TUTEL_INSTALLED:
            print("Tutel optimization requested but not installed. "
                  "Proceeding without Tutel.")

        if self.router_type == 'topk':
            if args.moe_topk == 1:
                self.moe_execution_func = self._top1_execution
            elif args.moe_topk == 2:
                self.moe_execution_func = self._top2_execution
        elif self.router_type == 'expert_choice':
            self.moe_execution_func = self._ec_execution

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:
        return self.moe_execution_func(input)

    def dispatch_expert_combine(self, dispatched_input, model_dim):
        # token dispatching
        dispatched_input = self._before_dispatch_a2a_in_tp(dispatched_input)
        dispatched_input = all_to_all(self.ep_group, dispatched_input)
        dispatched_input = self._after_dispatch_a2a_in_tp(dispatched_input)
        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(self.ep_size,
                                                    self.num_local_experts,
                                                    -1, model_dim)
        expert_output = self.experts(dispatched_input)                   
        expert_output = self._before_combine_a2a_in_tp(expert_output)
        # token combining
        expert_output = all_to_all(self.ep_group, expert_output)
        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(self.ep_size * self.num_local_experts,
                                              -1, model_dim)
        expert_output = self._after_combine_a2a_in_tp(expert_output)
        return expert_output

    # TODO: remove tutel code
    def _top1_execution(self, input):
        d_model = input[0].shape[-1]
        reshaped_input = input[0].reshape(-1, d_model)
        if self.use_tutel:
            l_aux, C, E, indices_, locations_, gates_ = self.gate(reshaped_input, input[1], True)
            S, M = reshaped_input.size(0), reshaped_input.size(1)
            if not hasattr(self, '_tutel_dispatcher'):
                self._tutel_dispatcher = tutel_moe.fast_dispatcher(
                    E, C, M, dispatch_dtype=reshaped_input.dtype)
            self._tutel_dispatcher.update(indices_, locations_, gates_, capacity=C)
            dispatched_input = self._tutel_dispatcher.encode(reshaped_input)
            # Reshape tutel's output from [e*c,m] to [e,c,m]
            dispatched_input = dispatched_input.reshape(self.ep_size * self.num_local_experts,
                                            -1, d_model)
        else:
            l_aux, gating, indices, mask = self.gate(reshaped_input, input[1])
            masked_reshaped_input = reshaped_input * (mask.sum(axis=1).unsqueeze(1))
            dispatched_input = masked_reshaped_input.index_select(
                dim=0, index=indices.view(-1)).reshape(self.ep_size * self.num_local_experts, -1, d_model)
        
        expert_output = self.dispatch_expert_combine(dispatched_input, d_model)
        if self.use_tutel:
            combined_output = self._tutel_dispatcher.decode(expert_output.view(E * C, M))
        else:
            combined_output = torch.einsum("ec,ecm->ecm", gating.type_as(input[0]), expert_output)
            combined_output = torch.scatter_add(
                torch.zeros_like(reshaped_input), 0,
                indices.view(-1, 1).expand(-1, reshaped_input.shape[1]),
                combined_output.reshape(-1, d_model))
        acts = combined_output.reshape(input[0].shape)
        # Use an autograd function to activate the backward computation for l_aux
        acts = AuxLossBackwardHook.apply(acts, l_aux)
        return acts

    def _top2_execution(self, input):
        d_model = input[0].shape[-1]
        reshaped_input = input[0].reshape(-1, d_model)

        l_aux, combine_weights, dispatch_mask, self.exp_counts = self.gate(reshaped_input, input[1])
        dispatched_input = einsum("sec,sm->ecm",
                                  dispatch_mask.type_as(input[0]),
                                  reshaped_input)

        expert_output = self.dispatch_expert_combine(dispatched_input, d_model)
        combined_output = einsum("sec,ecm->sm",
                                 combine_weights.type_as(input[0]),
                                 expert_output)

        acts = combined_output.reshape(input[0].shape)
        # Use an autograd function to activate the backward computation for l_aux
        acts = AuxLossBackwardHook.apply(acts, l_aux)
        return acts

    def _ec_execution(self, input):
        d_model = input[0].shape[-1]
        reshaped_input = input[0].reshape(-1, d_model)
            
        l_aux, gating, indices = self.gate(reshaped_input, input[1])
        dispatched_input = reshaped_input.index_select(dim=0, index=indices.view(-1)).reshape(
            self.ep_size * self.num_local_experts, -1, d_model)
            
        expert_output = self.dispatch_expert_combine(dispatched_input, d_model)

        combined_output = torch.einsum("ec,ecm->ecm", gating.type_as(input[0]), expert_output)
        combined_output = torch.scatter_add(
            torch.zeros_like(reshaped_input), 0,
            indices.view(-1, 1).expand(-1, reshaped_input.shape[1]),
            combined_output.reshape(-1, d_model))
        acts = combined_output.reshape(input[0].shape)
        return acts

    def _before_dispatch_a2a_in_tp(self, dispatched_input):
        args = get_args()
        if args.tensor_model_parallel_size <= 1:
            return dispatched_input
        if self.expert_tensor_parallelism:
            # Expert Tensor Parallel
            # No operation in the forward pass and all-reduce in the backward pass
            dispatched_input = copy_to_tensor_model_parallel_region(dispatched_input)
            if args.moe_input_feature_slicing:
                dispatched_input = scatter_to_tensor_model_parallel_region(dispatched_input)
        else:
            # EP w/o SP
            # Drop duplicate tokens to ensure correctness.
            # Better enable SP with TP+EP as it no extra
            # communications, less memory footprint and better performance.
            if not args.sequence_parallel:
                dispatched_input = scatter_tokens_to_tensor_parallel_region(dispatched_input)
                
        return dispatched_input
    
    def _after_dispatch_a2a_in_tp(self, dispatched_input):
        args = get_args()
        if args.tensor_model_parallel_size <= 1:
            return dispatched_input
        if self.expert_tensor_parallelism and args.moe_input_feature_slicing:
            dispatched_input = gather_from_tensor_model_parallel_region(dispatched_input)
        return dispatched_input
    
    def _before_combine_a2a_in_tp(self, expert_output):
        args = get_args()
        if args.tensor_model_parallel_size <= 1:
            return expert_output
        # if self.expert_tensor_parallelism and args.moe_input_feature_slicing:
        #     expert_output = scatter_to_tensor_model_parallel_region(expert_output)
        return expert_output
    
    def _after_combine_a2a_in_tp(self, expert_output):
        args = get_args()
        if args.tensor_model_parallel_size <= 1:
            return expert_output
        # if self.expert_tensor_parallelism and args.moe_input_feature_slicing:
        #     expert_output = gather_from_tensor_model_parallel_region(expert_output)
        if not self.expert_tensor_parallelism and not args.sequence_parallel:
            # The dropped duplicate tokens need to be gathered on each
            # tensor parallel rank again for the tensor-parallel
            # non-expert of the next layer.
            expert_output = gather_tokens_from_tensor_parallel_region(expert_output)
        return expert_output

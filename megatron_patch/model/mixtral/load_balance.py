from collections import defaultdict
from copy import deepcopy
from typing import List, Optional, Tuple

import megatron
import torch
import torch.distributed as dist
from megatron.core.parallel_state import get_tensor_and_expert_parallel_group, get_expert_model_parallel_rank, \
    get_tensor_model_parallel_rank, get_data_parallel_group, \
    get_data_modulo_expert_parallel_group
from megatron.training import get_args
from torch import nn
from torch.distributed import ProcessGroup


class LoadBalancer():
    def __init__(self, experts, router) -> None:
        self.args = get_args()
        self.experts = experts
        self.router = router
        self.tp_ep_group = get_tensor_and_expert_parallel_group()
        self.dp_group = get_data_parallel_group()
        self.dp_ep_group = get_data_modulo_expert_parallel_group()
        self.local_load = None
        self.ep_rank = get_expert_model_parallel_rank()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_ep_rank = get_expert_model_parallel_rank()
        self.rank = dist.get_rank()
        # build relation between device and expert
        self.tp_size = self.args.tensor_model_parallel_size
        self.ep_size = self.args.expert_model_parallel_size
        self.expert_per_device = len(self.experts.local_experts)

        self.expert_placement = self._gen_expert_placement()

    @torch.no_grad()
    def print_token_dist(self, step):
        print(f"[print_token_dist] step:{step} layer_num:{getattr(self.experts.local_experts[0].linear_fc1.weight, 'layer_num',None)} rank:{self.rank} local_load:{self.local_load}", flush=True)
        if not hasattr(self.args, 'load_balance_interval'):
            self.local_load = None

    @torch.no_grad()
    def _gen_expert_placement(self,):
        """
        generate expert placement
        return: expert id , tp_rank, (global_rank, local_expert_id)
        """
        self.device_mesh = torch.tensor([self.rank, self.ep_rank, self.tp_rank],device=torch.cuda.current_device())
        self.device_meshs = [torch.tensor([0,0,0], dtype=torch.int64, device=torch.cuda.current_device()) for _ in range(self.tp_size * self.ep_size)]
        dist.all_gather(self.device_meshs, self.device_mesh, group=self.tp_ep_group)
        expert_placement = defaultdict(dict)

        for device_mesh in self.device_meshs:
            global_rank, ep_rank, tp_rank = device_mesh.tolist()
            for local_expert_id in range(self.expert_per_device):
                expert_placement[ep_rank*self.expert_per_device+local_expert_id][tp_rank] = (global_rank, local_expert_id)
        return expert_placement

    @staticmethod
    def _get_diff_from_avg(data: List, group: int, avg: float) -> float:
        return abs(sum(data[group]) / len(data[group]) - avg)

    @staticmethod
    def _check_convergence(data: List, avg: float, tolerance: float):
        """
        Check whether the data is converged after swap.
        """
        for sublist in data:
            if abs(sum(sublist) / len(sublist) - avg) > tolerance * avg:
                return False
        return True

    @torch.no_grad()
    def _swap_expert_param_and_optim(
        self,
        weight: nn.Parameter,
        expert_info: list,
        comm_group: ProcessGroup,
        send_first: bool,
        comm_rank: int,
        optim,
    ):
        working_weight_ptr = weight
        from megatron.optimizer.optimizer import Float16OptimizerWithFloat16Params
        for opt in optim.chained_optimizers:
            if isinstance(opt, Float16OptimizerWithFloat16Params):
                fp16_param_optimezer = opt
        for param_group in optim.param_groups:
            for param in param_group['params']:
                if hasattr(param, 'expert_id'):

                if hasattr(param, 'expert_id') and param.expert_id == expert_info and torch.all((param.half() == weight.half()) == True):
                    # swap optimizer param
                    if not fp16_param_optimezer.state[param]:
                        raise IndexError("param not in fp16_param_optimezer.state")
                    for opti_param in fp16_param_optimezer.state[param]:
                        self._swap_expert_single_tensor(
                            fp16_param_optimezer.state[param][opti_param],
                            comm_group,
                            send_first,
                            comm_rank,
                        )

                    # swap fp32 main param
                    self._swap_expert_single_tensor(
                        param,
                        comm_group,
                        send_first,
                        comm_rank,
                    )
                    break

        # exchange weight
        self._swap_expert_single_tensor(
            working_weight_ptr,
            comm_group,
            send_first,
            comm_rank,
        )

    @staticmethod
    def _swap_expert_single_tensor(
        weight: nn.Parameter,
        comm_group: ProcessGroup,
        send_first: bool,
        comm_rank: int,
    ):
        # exchange weight
        local_weight = weight.data
        new_weight = torch.empty_like(local_weight)
        if send_first:
            dist.send(local_weight, dst=comm_rank, group=comm_group)
            dist.recv(new_weight, src=comm_rank, group=comm_group)
        else:
            dist.recv(new_weight, src=comm_rank, group=comm_group)
            dist.send(local_weight, dst=comm_rank, group=comm_group)

        weight.data.copy_(new_weight)

    @torch.no_grad()
    def gen_swap_list(self):
        # all_reduce expert load
        tmp = self.local_load.cuda()
        all_load = [torch.zeros_like(tmp) for _ in range(self.tp_size*self.ep_size)]
        dist.all_gather(all_load, tmp, group=self.tp_ep_group)

        #all reduce load count across dp ep group
        group_size = dist.get_world_size(self.dp_ep_group)
        if group_size > 1:
            all_load = torch.stack(all_load)
            dist.all_reduce(all_load, group=self.dp_ep_group)

        all_load = all_load[::self.tp_size]
        self.local_load = None
        swap_list = self._search_balance([i.tolist() for i in all_load])
        result = []
        for swap in swap_list:
            source_device, source_id, target_device, target_id = swap
            source_expert_id = len(all_load[0]) * source_device + source_id
            target_expert_id = len(all_load[0]) * target_device + target_id
            result.append([source_expert_id, target_expert_id])
        return result

    @staticmethod
    def _normalize_data(data: List) -> List:
        max_value = max(max(sublist) for sublist in data)
        data = [[i / max_value for i in sublist] for sublist in data]
        return data

    @staticmethod
    def _swap_data(data: List, group_i: int, index_i: int, group_j: int, index_j: int) -> None:
        data[group_i][index_i], data[group_j][index_j] = (
            data[group_j][index_j],
            data[group_i][index_i],
        )

    @torch.no_grad()
    def _beam_search(
        self,
        inputs: Tuple[List, float, List],
        beam_width: int,
        avg: float,
        group_swap_factor: float,
    ) -> List:
        """
        Beam search for the best swap combination.
        Specifically, we swap two elements from two groups and calculate the score.
        The score is the difference between the origin group sum and the new group sum.
        The larger the score, the better the swap combination.

        Args:
            inputs (Tuple): (data, origin_score, swap_list)
            beam_width (int): beam width for beam search
            avg (float): average value of the data
            group_swap_factor (float): group loss for group swap loss

        Returns:
            List: results list
        """
        data, origin_score, swap_list = inputs
        results = []
        group_num = len(data)
        group_size = len(data[0])
        origin_diff_list = [self._get_diff_from_avg(data, i, avg) for i in range(group_num)]

        for group_num_i in range(group_num):
            for group_size_i in range(group_size):
                for group_num_j in range(group_num_i + 1, group_num):
                    for group_size_j in range(group_size):
                        new_data = deepcopy(data)
                        # calculate origin group sum
                        origin_diff = origin_diff_list[group_num_i] + origin_diff_list[group_num_j]
                        # swap data
                        self._swap_data(
                            new_data,
                            group_num_i,
                            group_size_i,
                            group_num_j,
                            group_size_j,
                        )
                        # calculate new group sum
                        new_diff = self._get_diff_from_avg(new_data, group_num_i, avg) + self._get_diff_from_avg(
                            new_data, group_num_j, avg
                        )
                        # caculate score
                        new_score = origin_diff - new_diff
                        if new_score > 0:
                            new_score = origin_score + new_score
                            # get swap loss
                            swap_loss = self._get_swap_loss(
                                group_swap_factor,
                                swap_list,
                                group_num_i,
                                group_size_i,
                                group_num_j,
                                group_size_j,
                            )
                            new_score = new_score - swap_loss
                            # update swap list
                            new_swap_list = swap_list + [(group_num_i, group_size_i, group_num_j, group_size_j)]
                            results.append((new_data, new_score, new_swap_list))
        # sort results
        results.sort(key=lambda x: x[1], reverse=True)
        # select top k results
        results = results[:beam_width]
        return results

    @staticmethod
    def _get_swap_loss(
        group_swap_factor: float,
        swap_list: List,
        group_i: int,
        index_i: int,
        group_j: int,
        index_j: int,
    ) -> float:
        """
        Get swap loss. The swap loss is used to avoid the situation that
        the same index is swapped twice and the same group is swapped for multiple times.
        """
        swap_loss = 0
        for swap in swap_list:
            for group_id, index_id in zip([group_i, group_j], [index_i, index_j]):
                # the group has been swapped
                if group_id in [swap[0], swap[2]]:
                    # the index has been swapped
                    # we want to avoid the situation that the same index is swapped twice
                    if index_id in [swap[1], swap[3]]:
                        swap_loss += 1e5
                    # the index has not been swapped
                    # this is acceptable but as less as possible
                    else:
                        swap_loss += group_swap_factor
        return swap_loss

    @torch.no_grad()
    def _search_balance(
        self,
        data: List,
        tolerance: Optional[float] = 0.1,
        beam_width: Optional[int] = 8,
        group_swap_factor: Optional[float] = 0.4,
        return_swapped_data: Optional[bool] = False,
    ) -> Tuple[List, List]:
        """
        Search for the best swap combination to balance the data within the specified tolerance.
        And return the balanced data and the swap list. The swap list is used to record the swap.
        The swap list is a list of tuples. Each tuple is a swap operation.

        Args:
            data (List): expert load list.
                E.g. [[9.2, 8.3], [2.3, 10.0], [6.1, 7.2], [5.3, 3.2]]
                This means there are 4 devices and each devices has 2 experts.
                The value is the load of the expert.
            tolerance (float): tolerance for balance.
            beam_width (int): beam width for beam search.
            group_swap_factor (float): group swap factor for group swap loss.
                The bigger it is, the less times a group will be swapped.
            return_swapped_data (bool): whether to return the swapped data.

        Returns:
            Tuple: (balanced data, swap list).
                The swap list is a list of tuples. Each tuple is a swap operation.
                E.g. [(0, 0, 1, 0), (...), (...)]. The first tuple means
                the first expert of the first device is swapped with the first expert
                of the second device.
        """
        norm_data = self._normalize_data(data)
        avg = sum(sum(sublist) / len(sublist) for sublist in norm_data) / len(norm_data)
        results = [(norm_data, 0, [])]
        stop_flag = False

        while stop_flag == False:
            new_results = []
            best_score = results[0][1]
            for i in range(len(results)):
                new_results.extend(self._beam_search(results[i], beam_width, avg, group_swap_factor))
            if len(new_results) == 0:
                stop_flag = True
                break
            new_results.sort(key=lambda x: x[1], reverse=True)
            new_best_score = new_results[0][1]
            if new_best_score == best_score:
                stop_flag = True
                break
            new_results = new_results[:beam_width]
            results = new_results
            for i in results:
                if self._check_convergence(results[0][0], avg, tolerance):
                    stop_flag = True
                    break

        swap_list = results[0][2]
        if return_swapped_data:
            out = deepcopy(data)
            for swap in swap_list:
                self._swap_data(out, *swap)
            return out, swap_list
        else:
            return swap_list


    @torch.no_grad()
    def swap_model_param(self, swap_list, optim):

        local_rank = dist.get_rank()
        weight_list = []
        for i in self.experts.local_experts:
            weight_list.append([i.linear_fc1.weight, i.linear_fc2.weight])

        def swap_router_param(source_expert_id, target_expert_id):
            def _swap_router_param(router_param):
                router_param_tensor = list(torch.split(router_param.clone().detach(), self.args.num_experts))
                router_param_tensor[source_expert_id], router_param_tensor[target_expert_id] = router_param_tensor[target_expert_id], router_param_tensor[source_expert_id]
                router_param_tensor = torch.concat(router_param_tensor)
                router_param.data.copy_(router_param_tensor)
            return _swap_router_param
        for swap in swap_list:
            source_expert_id, target_expert_id = swap
            # swap router param and router optimizer param
            origin_source = self.router.weight.data[source_expert_id].clone().detach()
            origin_target = self.router.weight.data[target_expert_id].clone().detach()
            self.router.weight.data[source_expert_id], self.router.weight.data[target_expert_id] = (
                origin_target,
                origin_source,
            )

            for opt in optim.chained_optimizers:
                if isinstance(opt, megatron.optimizer.distrib_optimizer.DistributedOptimizer):
                    distrubute_optim = opt

            self._swap_main_params_to_model_params(distrubute_optim, self.router.weight, swap_router_param(source_expert_id, target_expert_id))

            # swap expert and expert optimizer param
            for tp_rank in range(self.tp_size):
                source_global_rank, source_expert_index = self.expert_placement[source_expert_id][tp_rank]
                target_global_rank, target_expert_index = self.expert_placement[target_expert_id][tp_rank]

                if local_rank == source_global_rank:
                    tmp_expert_index = source_expert_index
                elif local_rank == target_global_rank:
                    tmp_expert_index = target_expert_index
                else:
                    continue

                for fc_id, weight in enumerate(weight_list[tmp_expert_index]):
                    if local_rank == source_global_rank:
                        self._swap_expert_param_and_optim(
                            weight,
                            [fc_id, source_expert_index],
                            self.tp_ep_group,
                            True,
                            target_global_rank,
                            optim,
                        )
                    else:
                        self._swap_expert_param_and_optim(
                            weight,
                            [fc_id, target_expert_index],
                            self.tp_ep_group,
                            False,
                            source_global_rank,
                            optim,
                        )


    @torch.no_grad()
    def update_load(self, tokens_per_expert):
        if self.local_load != None:
            self.local_load += tokens_per_expert
        else:
            self.local_load = tokens_per_expert


    @torch.no_grad()
    def balance_load(self, optim):
        swap_list = self.gen_swap_list()
        print(f"[swap info] rank:{self.rank} swap_list:{swap_list}", flush=True)
        self.swap_model_param(swap_list, optim)

    @torch.no_grad()
    def _swap_main_params_to_model_params(self, optim, param, fn):
        """
        Copy main params to model params.

        Since this step is followed by an all-gather through the DDP's grad
        buffer, this method is responsible for copying the updated params
        from the main shards into the correct position in the grad buffer.
        """

        # Utility method for copying group params.
        def copy_group_params(shard_main_groups, model_groups, param):
            for shard_main_group, model_group in zip(shard_main_groups, model_groups):
                for shard_main_param, model_param in zip(shard_main_group, model_group):
                    # find target model param
                    if not (model_param.shape == param.shape and torch.all((model_param.half() == param.half()) == True)):
                        continue

                    fn(shard_main_param)
                    param_range_map = optim.get_model_param_range_map(model_param)
                    world_range = param_range_map["gbuf_world_in_bucket"]

                    assert world_range.size == shard_main_param.nelement()

                    gbuf_index, dtype, bucket_id = optim.model_param_gbuf_map[model_param]
                    model_param_buffer = optim.param_buffers[gbuf_index][bucket_id]

                    shard_model_param = model_param_buffer.view(-1)[
                        world_range.start : world_range.end
                    ]
                    shard_model_param.data.copy_(shard_main_param)

                    fn(optim.state[shard_main_param]['exp_avg'])
                    fn(optim.state[shard_main_param]['exp_avg_sq'])

        # Copy shard groups to model groups.
        copy_group_params(optim.shard_fp32_from_float16_groups, optim.model_float16_groups, param)
        copy_group_params(optim.shard_fp32_groups, optim.model_fp32_groups, param)
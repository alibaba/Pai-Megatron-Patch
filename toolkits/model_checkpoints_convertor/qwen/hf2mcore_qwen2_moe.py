# Copyright (c) 2025 Alibaba PAI Team.
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

import gc
import math
import safetensors.torch
import sys
import os
import re
import torch
import warnings
from collections import defaultdict
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2MoeForCausalLM,
)

from megatron.training.initialize import initialize_megatron
from megatron.training import get_args
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, read_metadata

path_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.append(os.path.join(path_dir, "examples"))
from qwen2.pretrain_qwen2_moe import model_provider
from megatron_patch.arguments import get_patch_args

from toolkits.model_checkpoints_convertor.utils import (
    save_state_dict,
    save_hfmodel
)

from megatron.core.models.gpt import GPTModel


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

warnings.filterwarnings("ignore", category=UserWarning)


def generate_rank_group(
        tensor_model_parallel_size,
        expert_tensor_parallel_size,
        expert_model_parallel_size,
        pipeline_model_parallel_size
):
    """
        copy from toolkits/model_checkpoints_convertor/deepseek/hf2mcore_deepseek_v3_moe.py

        This function attempts to generate group rank on the minimal practicable world_size.
        Support Decoder-Only model currently.
    """
    tp, etp, ep, pp = (
        tensor_model_parallel_size,
        expert_tensor_parallel_size,
        expert_model_parallel_size,
        pipeline_model_parallel_size
    )
    minimal_worldsize = pp * math.lcm(tp, etp * ep)
    print(f"The given parallel config should be run on at least {minimal_worldsize} cards")
    dp = minimal_worldsize // (pp * tp)
    edp = minimal_worldsize // (pp * ep * etp)
    # NOTE: If user want to scale up cp_size, he should downscale
    # dp_size or scale up world_size, i.e., edp_size
    cp = 1

    # TODO: support other orders
    order = "tp-cp-ep-dp-pp"
    # In training:
    # Dense:
    # global_rank = tp_rank + cp_rank * tp_size + dp_rank * cp_size * tp_size + pp_rank * dp_size * cp_size * tp_size
    # MoE:
    # global_rank = etp_rank + ep_rank * etp_size + edp_rank * ep_size * etp_size + pp_rank * edp_size * ep_size * etp_size

    # In ckpt loading, each rank will load a checkpoint according to its (tp_rank, pp_rank, ep_rank)
    # Thus, (tp_rank, ep_rank) should map to a unique etp_rank
    rank_mappings = dict()
    local_ids = []
    for global_rank in range(minimal_worldsize):
        tp_rank = global_rank % tp
        etp_rank = global_rank % etp
        ep_rank = (global_rank // etp) % ep
        pp_rank = global_rank // (dp * tp)

        if (tp_rank, ep_rank) not in rank_mappings:
            rank_mappings[(tp_rank, ep_rank)] = etp_rank

        if rank_mappings[(tp_rank, ep_rank)] != etp_rank:
            raise ValueError("The legacy checkpoint format cannot support this parallel config.")

        local_ids.append((tp_rank, etp_rank, ep_rank, pp_rank))
    return local_ids


def add_model_args(parser):

    parser.add_argument(
        "--target-tensor-model-parallel-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--target-pipeline-model-parallel-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--target-expert-model-parallel-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--target-expert-tensor-parallel-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--target-decoder-first-pipeline-num-layers",
        type=int,
        default=None
    )

    parser.add_argument(
        "--hf-ckpt-path",
        type=str
    )

    parser.add_argument(
        "--save-safetensors",
        action='store_false',
    )

    return parser


def load_megatron_model(args):
    os.makedirs(args.save, exist_ok=True)
    os.system("cp -rf " + args.hf_ckpt_path + "/*config.json " + args.save)
    os.system("cp -rf " + args.hf_ckpt_path + "/tokenizer* " + args.save)
    # os.system("cp -rf " + args.hf_ckpt_path + "/*.py " + args.save)
    os.system("cp -rf " + args.hf_ckpt_path + "/merges.txt " + args.save)

    os.system("cp -rf " + args.hf_ckpt_path + "/*config.json " + args.load)
    os.system("cp -rf " + args.hf_ckpt_path + "/tokenizer* " + args.load)
    # os.system("cp -rf " + args.hf_ckpt_path + "/*.py " + args.load)
    os.system("cp -rf " + args.hf_ckpt_path + "/vocab.json " + args.load)
    os.system("cp -rf " + args.hf_ckpt_path + "/merges.txt " + args.load)
    model = model_provider()


    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size
    args.expert_tensor_parallel_size = args.target_expert_tensor_parallel_size

    if args.num_experts is not None:
        args.expert_model_parallel_size = args.target_expert_model_parallel_size

    if args.tensor_model_parallel_size > 1:
        args.sequence_parallel = True

    model_path = args.load
    tracker_filename = get_checkpoint_tracker_filename(model_path)
    iteration, release = read_metadata(tracker_filename)
    head_dim = args.hidden_size // args.num_attention_heads if args.kv_channels is None else args.kv_channels

    group_per_split = args.num_query_groups // args.tensor_model_parallel_size

    if args.num_experts is not None:
        num_local_experts = args.num_experts // args.expert_model_parallel_size
    state_dict = {}
    mid_state = defaultdict(list)

    if (
        args.tensor_model_parallel_size >= 1
        and args.pipeline_model_parallel_size >= 1
        and args.expert_model_parallel_size >= 1
        and args.num_experts % args.expert_model_parallel_size == 0
        and args.expert_tensor_parallel_size == 1
    ):
        if args.target_decoder_first_pipeline_num_layers is not None:
            remained_layers = args.num_layers - args.target_decoder_first_pipeline_num_layers
            remained_stages = args.pipeline_model_parallel_size - 1
            assert remained_layers % remained_stages == 0
            pp_layers_per_stage = [args.target_decoder_first_pipeline_num_layers] +([remained_layers // remained_stages] * remained_stages)
        else:
            pp_layers_per_stage = [args.num_layers // args.pipeline_model_parallel_size] * args.pipeline_model_parallel_size

        layers_to_copy = {}
        for tp_rank in range(args.tensor_model_parallel_size):
            for ep_rank in range(tp_rank, args.expert_model_parallel_size, args.tensor_model_parallel_size):
                for pp_rank in range(args.pipeline_model_parallel_size):
                    layer_offset = sum(pp_layers_per_stage[:pp_rank])
                    for layer in range(pp_layers_per_stage[pp_rank]):
                        pp_layer_id = layer + layer_offset
                        layers_to_copy[(pp_rank, layer)] = pp_layer_id

                    if args.expert_model_parallel_size > 1:
                        checkpoint_name = get_checkpoint_name(model_path, iteration, release, True, tp_rank, pp_rank, True,
                                                              ep_rank)
                    elif args.expert_model_parallel_size == 1:
                        checkpoint_name = get_checkpoint_name(model_path, iteration, release, True, tp_rank, pp_rank,
                                                              False)
                    print(f'load {checkpoint_name}')
                    split_state = torch.load(checkpoint_name, map_location="cpu", weights_only=False)['model']
                    for k, v in split_state.items():
                        if "_extra_state" in k:
                            continue
                        try:
                            if 'experts' in k and "shared_experts" not in k:
                                pattern = r'weight(\d+)'
                                local_expert_rank = int(re.findall(pattern, k)[0])
                                expert_rank = local_expert_rank + num_local_experts * ep_rank
                                k = k.replace(f'weight{local_expert_rank}', f'weight{expert_rank}')
                            pattern = re.compile(r'\d+')
                            res = pattern.findall(k)
                            tgt = re.sub(r"decoder.layers.\d+", "decoder.layers." + str(layers_to_copy[(pp_rank, int(res[0]))]), k)
                            if 'linear_proj' in k or 'shared_experts.linear_fc1' in k or 'shared_experts.linear_fc2' in k or \
                                "linear_qkv" in k:
                                if ep_rank == tp_rank:
                                    mid_state[tgt].append(v)
                            else:
                                mid_state[tgt].append(v)
                        except:
                            if "word_embeddings" in k:
                                if ep_rank == tp_rank and pp_rank == 0:
                                    mid_state[k].append(v)
                            elif "output_layer" in k or "final_layernorm" in k:
                                if ep_rank == tp_rank and pp_rank == args.pipeline_model_parallel_size - 1:
                                    mid_state[k].append(v)
                            else:
                                raise ValueError(f"{k} is missing!! ")

        for k, v in mid_state.items():
            if 'extra_state' in k:
                continue
            elif not isinstance(v[0], torch.Tensor) or 'router' in k or 'gate' in k:
                target_v = v[0]
            elif 'word_embeddings' in k or 'output_layer' in k:
                target_v = torch.cat(v, dim=0)
            elif 'linear_proj' in k:
                target_v = torch.cat(v, dim=1)
            elif 'linear_qkv.weight' in k:
                viewed = [x.view(group_per_split, -1, head_dim, args.hidden_size) for x in v]
                target_v = torch.cat(viewed, dim=0).view(-1, args.hidden_size)
            elif 'linear_qkv.bias' in k:
                viewed = [x.view(group_per_split, -1) for x in v]
                target_v = torch.cat(viewed, dim=0).view(-1)
            elif "experts.linear_fc2" in k and "shared_experts" not in k:
                target_v = v[0]
            elif 'experts.linear_fc1' in k and "shared_experts" not in k:
                target_v = v[0]
            elif 'shared_experts.linear_fc1' in k:
                viewed = [x.view(2, -1, args.hidden_size) for x in v]
                target_v = torch.cat(viewed, dim=1).view(-1, args.hidden_size)
            elif "shared_experts.linear_fc2" in k:
                target_v = torch.cat(v, dim=1)
            elif "shared_experts.gate_weight" in k or 'layer_norm_weight' in k or 'pre_mlp_layernorm' in k or 'final_layernorm' in k or 'q_layernorm' in k or 'k_layernorm' in k:
                target_v = v[0]
            else:
                raise ValueError(f"{k} is missing!")
            state_dict[k] = target_v

    elif (
        args.tensor_model_parallel_size >= 1
        and args.pipeline_model_parallel_size >= 1
        and args.expert_model_parallel_size >= 1
        and args.num_experts % args.expert_model_parallel_size == 0
        and args.expert_tensor_parallel_size > 1
    ):
        if args.target_decoder_first_pipeline_num_layers is not None:
            remained_layers = args.num_layers - args.target_decoder_first_pipeline_num_layers
            remained_stages = args.pipeline_model_parallel_size - 1
            assert remained_layers % remained_stages == 0
            pp_layers_per_stage = [args.target_decoder_first_pipeline_num_layers] +([remained_layers // remained_stages] * remained_stages)
        else:
            pp_layers_per_stage = [args.num_layers // args.pipeline_model_parallel_size] * args.pipeline_model_parallel_size

        layers_to_copy = {}
        for tp_rank in range(args.tensor_model_parallel_size):
            for ep_rank in range(args.expert_model_parallel_size):
                for pp_rank in range(args.pipeline_model_parallel_size):
                    layer_offset = sum(pp_layers_per_stage[:pp_rank])
                    for layer in range(pp_layers_per_stage[pp_rank]):
                        pp_layer_id = layer + layer_offset
                        layers_to_copy[(pp_rank, layer)] = pp_layer_id

                    if args.expert_model_parallel_size > 1:
                        checkpoint_name = get_checkpoint_name(model_path, iteration, release, True, tp_rank, pp_rank, True,
                                                              ep_rank)
                    elif args.expert_model_parallel_size == 1:
                        checkpoint_name = get_checkpoint_name(model_path, iteration, release, True, tp_rank, pp_rank,
                                                              False)
                    print(f'load {checkpoint_name}')
                    split_state = torch.load(checkpoint_name, map_location="cpu", weights_only=False)['model']
                    for k, v in split_state.items():
                        if "_extra_state" in k:
                            continue
                        try:
                            if 'experts' in k and "shared_experts" not in k:
                                pattern = r'weight(\d+)'
                                local_expert_rank = int(re.findall(pattern, k)[0])
                                expert_rank = local_expert_rank + num_local_experts * ep_rank
                                k = k.replace(f'weight{local_expert_rank}', f'weight{expert_rank}')
                            pattern = re.compile(r'\d+')
                            res = pattern.findall(k)
                            tgt = re.sub(r"decoder.layers.\d+", "decoder.layers." + str(layers_to_copy[(pp_rank, int(res[0]))]), k)
                            if 'linear_proj' in k or 'shared_experts.linear_fc1' in k or 'shared_experts.linear_fc2' in k or \
                                "linear_qkv" in k:
                                if ep_rank == 0:
                                    mid_state[tgt].append(v)
                            else:
                                mid_state[tgt].append(v)
                        except:
                            if "word_embeddings" in k:
                                if ep_rank == 0 and pp_rank == 0:
                                    mid_state[k].append(v)
                            elif "output_layer" in k or "final_layernorm" in k:
                                if ep_rank == 0 and pp_rank == args.pipeline_model_parallel_size - 1:
                                    mid_state[k].append(v)
                            else:
                                raise ValueError(f"{k} is missing!! ")

        for k, v in mid_state.items():
            if 'extra_state' in k:
                continue
            elif not isinstance(v[0], torch.Tensor) or 'router' in k or 'gate' in k:
                target_v = v[0]
            elif 'word_embeddings' in k or 'output_layer' in k:
                target_v = torch.cat(v, dim=0)
            elif 'linear_proj' in k:
                target_v = torch.cat(v, dim=1)
            elif 'linear_qkv.weight' in k:
                viewed = [x.view(group_per_split, -1, head_dim, args.hidden_size) for x in v]
                target_v = torch.cat(viewed, dim=0).view(-1, args.hidden_size)
            elif 'linear_qkv.bias' in k:
                viewed = [x.view(group_per_split, -1) for x in v]
                target_v = torch.cat(viewed, dim=0).view(-1)
            elif 'linear_fc1' in k:
                viewed = [x.view(2, -1, args.hidden_size) for x in v]
                target_v = torch.cat(viewed, dim=1).view(-1, args.hidden_size)
            elif "linear_fc2" in k:
                target_v = torch.cat(v, dim=1)
            elif "shared_experts.gate_weight" in k or 'layer_norm_weight' in k or 'pre_mlp_layernorm' in k or 'final_layernorm' in k or 'q_layernorm' in k or 'k_layernorm' in k:
                target_v = v[0]
            else:
                raise ValueError(f"{k} is missing!")
            state_dict[k] = target_v

    else: 
        raise ValueError('not support yet')

    model.load_state_dict(state_dict, strict=False)
    return model



def convert_checkpoint_from_megatron_to_transformers(mgmodel, hfmodel, args):

    if args.fp16:
        mgmodel = mgmodel.half()
        hfmodel = hfmodel.half()
    elif args.bf16:
        mgmodel = mgmodel.bfloat16()
        hfmodel = hfmodel.bfloat16()

    num_query_groups = args.num_query_groups
    hidden_size = args.hidden_size
    head_dim = hidden_size // args.num_attention_heads if args.kv_channels is None else args.kv_channels
    use_te = args.transformer_impl == "transformer_engine"
    value_num_per_group = args.num_attention_heads // num_query_groups  # 28//4
    q_dim_per_group = hidden_size // num_query_groups
    kv_dim_per_group = head_dim
    with torch.no_grad():
        hfmodel.model.embed_tokens.weight.copy_(mgmodel.embedding.word_embeddings.weight)
        for mglayer, hflayer in zip(mgmodel.decoder.layers, hfmodel.model.layers):
            if use_te:
                hflayer.input_layernorm.weight.copy_(mglayer.self_attention.linear_qkv.layer_norm_weight)
            else:
                hflayer.input_layernorm.weight.copy_(mglayer.input_layernorm.weight)

            qkv_weight = mglayer.self_attention.linear_qkv.weight.view(num_query_groups, -1, head_dim, hidden_size)
            q_weight, k_weight, v_weight = torch.split(qkv_weight, split_size_or_sections=[value_num_per_group, 1, 1], dim=1)
            hflayer.self_attn.q_proj.weight.copy_(q_weight.reshape(-1, hidden_size))
            hflayer.self_attn.k_proj.weight.copy_(k_weight.reshape(-1, hidden_size))
            hflayer.self_attn.v_proj.weight.copy_(v_weight.reshape(-1, hidden_size))

            if args.add_qkv_bias:
                # NOTE: compatability for Qwen3-moe
                qkv_bias = mglayer.self_attention.linear_qkv.bias.view(num_query_groups, -1)
                q_bias, k_bias, v_bias = torch.split(qkv_bias, split_size_or_sections=[q_dim_per_group, kv_dim_per_group, kv_dim_per_group], dim=1)
                q_bias = q_bias.contiguous().view(-1)
                k_bias = k_bias.contiguous().view(-1)
                v_bias = v_bias.contiguous().view(-1)

                hflayer.self_attn.q_proj.bias.copy_(q_bias)
                hflayer.self_attn.k_proj.bias.copy_(k_bias)
                hflayer.self_attn.v_proj.bias.copy_(v_bias)
            
            if args.qk_layernorm:
                # NOTE: compatability for Qwen3-moe
                hflayer.self_attn.q_norm.weight.copy_(mglayer.self_attention.q_layernorm.weight.data)
                hflayer.self_attn.k_norm.weight.copy_(mglayer.self_attention.k_layernorm.weight.data)

            hflayer.self_attn.o_proj.weight.copy_(mglayer.self_attention.linear_proj.weight)
            if args.num_experts is None:
                raise ValueError("num_experts is None")

            hflayer.mlp.gate.weight.copy_(mglayer.mlp.router.weight)
            for i, hfexpert in enumerate(hflayer.mlp.experts):
                linear_fc1_weighti = getattr(mglayer.mlp.experts.linear_fc1, 'weight' + str(i))
                linear_fc2_weighti = getattr(mglayer.mlp.experts.linear_fc2, 'weight' + str(i))
                gate_weight, up_weight = torch.split(linear_fc1_weighti,
                                                        split_size_or_sections=args.moe_ffn_hidden_size)
                hfexpert.gate_proj.weight.copy_(gate_weight)
                hfexpert.up_proj.weight.copy_(up_weight)
                hfexpert.down_proj.weight.copy_(linear_fc2_weighti)

            if args.moe_shared_expert_intermediate_size is not None:
                # NOTE: compatability for Qwen3-moe
                hflayer.mlp.shared_expert_gate.weight.copy_(mglayer.mlp.shared_experts.gate_weight)
                shared_expert_gate_weight, shared_expert_up_weight = \
                    torch.split(mglayer.mlp.shared_experts.linear_fc1.weight,
                                split_size_or_sections=args.moe_shared_expert_intermediate_size)
                hflayer.mlp.shared_expert.gate_proj.weight.copy_(shared_expert_gate_weight)
                hflayer.mlp.shared_expert.up_proj.weight.copy_(shared_expert_up_weight)
                hflayer.mlp.shared_expert.down_proj.weight.copy_(mglayer.mlp.shared_experts.linear_fc2.weight)

            hflayer.post_attention_layernorm.weight.copy_(mglayer.pre_mlp_layernorm.weight)

        hfmodel.model.norm.weight.copy_(mgmodel.decoder.final_layernorm.weight)
        if args.untie_embeddings_and_output_weights:
            hfmodel.lm_head.weight.copy_(mgmodel.output_layer.weight)


def convert_checkpoint_from_transformers_to_megatron(hfmodel: Qwen2MoeForCausalLM, mgmodel: GPTModel, args):

    if args.fp16:
        mgmodel = mgmodel.half()
    elif args.bf16:
        mgmodel = mgmodel.bfloat16()

    head_dim = hidden_size // args.num_attention_heads if args.kv_channels is None else args.kv_channels
    group_per_split = args.num_query_groups // args.target_tensor_model_parallel_size

    with torch.no_grad():
        mgmodel.embedding.word_embeddings.weight.copy_(hfmodel.model.embed_tokens.weight)
        num_query_groups = args.num_query_groups

        from tqdm import tqdm
        for layer_idx, (mglayer, hflayer) in tqdm(enumerate(zip(mgmodel.decoder.layers, hfmodel.model.layers)), total=len(mgmodel.decoder.layers)):

            mglayer.self_attention.linear_qkv.layer_norm_weight.copy_(hflayer.input_layernorm.weight)

            q_proj_weight = hflayer.self_attn.q_proj.weight.view(num_query_groups, -1, head_dim, args.hidden_size)
            k_proj_weight = hflayer.self_attn.k_proj.weight.view(num_query_groups, -1, head_dim, args.hidden_size)
            v_proj_weight = hflayer.self_attn.v_proj.weight.view(num_query_groups, -1, head_dim, args.hidden_size)
            qkv_proj = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=1).view(-1, args.hidden_size).contiguous()
            mglayer.self_attention.linear_qkv.weight.copy_(qkv_proj)

            if args.add_qkv_bias:
                # NOTE: compatability for Qwen3-moe
                q_proj_bias = hflayer.self_attn.q_proj.bias.view(num_query_groups, -1)
                k_proj_bias = hflayer.self_attn.k_proj.bias.view(num_query_groups, -1)
                v_proj_bias = hflayer.self_attn.v_proj.bias.view(num_query_groups, -1)
                qkv_bias = torch.cat([q_proj_bias, k_proj_bias, v_proj_bias], dim=1).view(-1).contiguous()
                mglayer.self_attention.linear_qkv.bias.copy_(qkv_bias)

            if args.qk_layernorm:
                # NOTE: compatability for Qwen3-moe
                mglayer.self_attention.q_layernorm.weight.copy_(hflayer.self_attn.q_norm.weight.data)
                mglayer.self_attention.k_layernorm.weight.copy_(hflayer.self_attn.k_norm.weight.data)

            mglayer.self_attention.linear_proj.weight.copy_(hflayer.self_attn.o_proj.weight)

            mglayer.mlp.router.weight.copy_(hflayer.mlp.gate.weight)
            for i, hf_expert in enumerate(hflayer.mlp.experts):
                fc1_weight = torch.cat([hf_expert.gate_proj.weight, hf_expert.up_proj.weight])
                linear_fc1_weighti = getattr(mglayer.mlp.experts.linear_fc1, 'weight' + str(i))
                linear_fc1_weighti.copy_(fc1_weight)
                linear_fc2_weighti = getattr(mglayer.mlp.experts.linear_fc2, 'weight' + str(i))
                linear_fc2_weighti.copy_(hf_expert.down_proj.weight)

            if args.moe_shared_expert_intermediate_size is not None:
                # NOTE: compatability for Qwen3-moe
                shared_fc1_weight = torch.cat(
                    [hflayer.mlp.shared_expert.gate_proj.weight, hflayer.mlp.shared_expert.up_proj.weight])
                mglayer.mlp.shared_experts.linear_fc1.weight.copy_(shared_fc1_weight)
                mglayer.mlp.shared_experts.linear_fc2.weight.copy_(hflayer.mlp.shared_expert.down_proj.weight)
                mglayer.mlp.shared_experts.gate_weight.data.copy_(hflayer.mlp.shared_expert_gate.weight)

            mglayer.pre_mlp_layernorm.weight.copy_(hflayer.post_attention_layernorm.weight)

        mgmodel.decoder.final_layernorm.weight.copy_(hfmodel.model.norm.weight)
        if args.untie_embeddings_and_output_weights:
            mgmodel.output_layer.weight.copy_(hfmodel.lm_head.weight)


def split_column_parallel(tensor, tp_rank, tp_size):
    seg = tensor.shape[0] // tp_size
    return tensor[seg * tp_rank: seg * (tp_rank + 1)]


def split_row_parallel(tensor, tp_rank, tp_size):
    seg = tensor.shape[1] // tp_size
    return tensor[:, seg * tp_rank: seg * (tp_rank + 1)]


def check_layer(layers_to_copy, k):
    pattern = re.compile(r"decoder.layers.\d+")
    res = pattern.findall(k)
    return res and res[0] in layers_to_copy.keys()


def save_mgmodel(mgmodel: GPTModel, args):

    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size
    args.expert_model_parallel_size = args.target_expert_model_parallel_size
    args.expert_tensor_parallel_size = args.target_expert_tensor_parallel_size

    if args.num_experts is not None:
        args.expert_model_parallel_size = args.target_expert_model_parallel_size

    os.makedirs(args.save, exist_ok=True)
    os.system("cp -rf " + args.load + "/*config.json " + args.save)
    os.system("cp -rf " + args.load + "/tokenizer* " + args.save)
    os.system("cp -rf " + args.load + "/merges.txt " + args.save)
    os.system("cp -rf " + args.load + "/vocab.json " + args.save)

    tracker_filepath = os.path.join(args.save, 'latest_checkpointed_iteration.txt')
    with open(tracker_filepath, "w") as f:
        f.write("release")

    head_dim = args.hidden_size // args.num_attention_heads if args.kv_channels is None else args.kv_channels
    group_per_split = args.num_query_groups // args.target_tensor_model_parallel_size

    full_model = mgmodel.state_dict_for_save_checkpoint()
    for k in list(full_model.keys()):
        if full_model[k] is None and '_extra_state' not in k:
            full_model.pop(k)
            continue
        if '_extra_state' in k and isinstance(full_model[k], torch.Tensor):
            full_model[k] = None

    if args.num_experts is not None:
        pattern = r'weight(\d+)'
        assert args.num_experts % args.expert_model_parallel_size == 0
        num_local_experts = args.num_experts // args.expert_model_parallel_size if args.num_experts else 0

    if args.target_decoder_first_pipeline_num_layers is not None:
        remained_layers = args.num_layers - args.target_decoder_first_pipeline_num_layers
        remained_stages = args.pipeline_model_parallel_size - 1
        assert remained_layers % remained_stages == 0
        pp_layers_per_stage = [ args.target_decoder_first_pipeline_num_layers] +([remained_layers // remained_stages] * remained_stages)
    else:
        pp_layers_per_stage = [args.num_layers // args.pipeline_model_parallel_size] * args.pipeline_model_parallel_size

    for (tp_rank, etp_rank, ep_rank, pp_rank) in generate_rank_group(
            args.tensor_model_parallel_size,
            args.expert_tensor_parallel_size,
            args.expert_model_parallel_size,
            args.pipeline_model_parallel_size
    ):
        model_split = {}
        layer_offset = sum(pp_layers_per_stage[:pp_rank])
        layers_to_copy = {}
        for layer in range(pp_layers_per_stage[pp_rank]):
            pp_layer_id = layer + layer_offset
            layers_to_copy[f"decoder.layers.{pp_layer_id}"] = layer
        checkpoint_name = get_checkpoint_name(
            args.save, 0, True,
            args.pipeline_model_parallel_size > 1,
            tp_rank,
            pp_rank,
            args.expert_model_parallel_size > 1,
            ep_rank
        )
        print(f'tensor_parallel & pipeline_parallel & expert_parallel, save model to {checkpoint_name}')
        for k, v in full_model.items():
            if check_layer(layers_to_copy, k):
                layer_pattern = re.compile(r'\d+')
                res = layer_pattern.findall(k)
                k = re.sub(r"decoder.layers.\d+", "decoder.layers." + str(layers_to_copy["decoder.layers." + res[0]]), k)
            elif not ("word_embeddings" in k or "output_layer" in k or "final_layernorm" in k):
                continue

            if not isinstance(v, torch.Tensor):
                target_v = v
            elif 'linear_qkv.weight' in k:
                viewed = v.view(args.num_query_groups, -1, head_dim, args.hidden_size)
                viewed = viewed[group_per_split * tp_rank: group_per_split * (tp_rank + 1)]
                target_v = viewed.view(-1, args.hidden_size)
            elif 'linear_qkv.bias' in k:
                viewed = v.view(args.num_query_groups, -1, head_dim)
                viewed = viewed[group_per_split * tp_rank: group_per_split * (tp_rank + 1)]
                target_v = viewed.view(-1)
            elif 'linear_proj' in k:
                seg = v.shape[1] // args.tensor_model_parallel_size
                target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]
            elif 'experts' in k and 'shared_experts' not in k:
                expert_rank = int(re.findall(pattern, k)[0])
                if expert_rank // num_local_experts != ep_rank:
                    continue
                expert_local_rank = expert_rank % num_local_experts
                k = k.replace(f'weight{expert_rank}', f'weight{expert_local_rank}')
                if 'linear_fc1' in k:
                    viewed = v.view(-1, args.moe_ffn_hidden_size, args.hidden_size)
                    seg = args.moe_ffn_hidden_size // args.expert_tensor_parallel_size
                    target_v = viewed[:, seg * etp_rank: seg * (etp_rank + 1), :].reshape(-1, args.hidden_size)
                elif 'linear_fc2' in k:
                    target_v = split_row_parallel(v, etp_rank, args.expert_tensor_parallel_size)
                else:
                    raise NotImplementedError
            elif 'shared_experts' in k and 'gate' not in k:
                if 'linear_fc1' in k:
                    viewed = v.view(-1, args.moe_shared_expert_intermediate_size,
                                    args.hidden_size)
                    seg = args.moe_shared_expert_intermediate_size // args.tensor_model_parallel_size
                    target_v = viewed[:, seg * tp_rank: seg * (tp_rank + 1), :].reshape(-1, args.hidden_size)
                elif 'linear_fc2' in k:
                    seg = v.shape[1] // args.tensor_model_parallel_size
                    target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]

            elif "word_embeddings" in k or "output_layer" in k:
                seg = v.shape[0] // args.tensor_model_parallel_size
                target_v = v[seg * tp_rank: seg * (tp_rank + 1)]
            else:
                target_v = v

            if "word_embeddings" in k:
                if pp_rank == 0:
                    model_split[k] = target_v
            elif "output_layer" in k or "final_layernorm" in k:
                if pp_rank == args.pipeline_model_parallel_size - 1:
                    model_split[k] = target_v
            else:
                model_split[k] = target_v
        save_state_dict(args, [model_split], checkpoint_name)

    print(f'megatron model is save to {args.save}')


def add_extra_args(parser):
    parser = get_patch_args(parser)
    parser = add_model_args(parser)
    return parser

@torch.inference_mode()
def main():
    initialize_megatron(extra_args_provider=add_extra_args)
    args = get_args()

    if args.convert_checkpoint_from_megatron_to_transformers:
        config = AutoConfig.from_pretrained(args.hf_ckpt_path, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_pretrained(args.hf_ckpt_path, trust_remote_code=True, torch_dtype=config.torch_dtype)
        mg_model = load_megatron_model(args)
        convert_checkpoint_from_megatron_to_transformers(mg_model, hf_model, args)
        del mg_model
        gc.collect()
        save_hfmodel(args, hf_model)
    else:
        config = AutoConfig.from_pretrained(args.load, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_pretrained(args.load, trust_remote_code=True, torch_dtype=config.torch_dtype)
        mg_model = model_provider()
        convert_checkpoint_from_transformers_to_megatron(hf_model, mg_model, args)
        del hf_model
        gc.collect()
        save_mgmodel(mg_model, args)


if __name__ == "__main__":
    main()
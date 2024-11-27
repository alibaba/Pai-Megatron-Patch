import os
import re
import json
import torch
import copy
import safetensors
from collections import defaultdict
from typing import List, Dict, Tuple
from transformers import (
    AutoConfig,
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
)

from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, shard_checkpoint, load_sharded_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, read_metadata
from functools import partial
from megatron.training.utils import get_ltor_masks_and_position_ids
import sys

path_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.append(os.path.join(path_dir, "examples"))
sys.path.append(os.path.join(path_dir, "examples/qwen2_vl"))
from qwen2_vl.pretrain_qwen import model_provider
from megatron_patch.arguments import get_patch_args

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

import numpy as np
from collections.abc import Mapping, Sequence
@torch.inference_mode()
def clone_state_dict(elem):
    """clone all tensors in the elem to cpu device.
    """
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        elem = elem.clone()
    elif isinstance(elem, (np.ndarray, str)):
        pass
    elif isinstance(elem, Mapping):
        elem = dict(elem)
        for k, v in elem.items():
            elem[k] = clone_state_dict(v)
        elem = elem_type(elem)
    elif isinstance(elem, Sequence):
        elem = list(elem)
        for i in range(len(elem)):
            elem[i] = clone_state_dict(elem[i])
        elem = elem_type(elem)
    return elem

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
        "--hf-ckpt-path",
        type=str
    )

    parser.add_argument(
        "--save-safetensors",
        action='store_false',
    )

    return parser


def name_to_expert_rank(key):
    pattern = r'local_experts\.(\d+)\.'
    expert_rank = int(re.findall(pattern, key)[0])
    return expert_rank

def load_megatron_model(args):
    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size

    if args.tensor_model_parallel_size >1:
        args.sequence_parallel = True

    assert args.num_query_groups >= args.target_tensor_model_parallel_size

    os.makedirs(args.save, exist_ok=True)
    os.system("cp -rf " + args.hf_ckpt_path + "/config*.json " + args.save)
    os.system("cp -rf " + args.hf_ckpt_path + "/generation_config.json " + args.save)
    os.system("cp -rf " + args.hf_ckpt_path+ "/tokenizer* " + args.save)
    os.system("cp -rf " + args.hf_ckpt_path + "/vocab.json " + args.save)
    os.system("cp -rf " + args.hf_ckpt_path + "/merges.txt " + args.save)

    os.system("cp -rf " + args.hf_ckpt_path + "/config*.json " + args.load)
    os.system("cp -rf " + args.hf_ckpt_path + "/generation_config.json " + args.load)
    os.system("cp -rf " + args.hf_ckpt_path+ "/tokenizer* " + args.load)
    os.system("cp -rf " + args.hf_ckpt_path + "/vocab.json " + args.load)
    os.system("cp -rf " + args.hf_ckpt_path + "/merges.txt " + args.load)

    model = model_provider()


    model_path = args.load
    tracker_filename = get_checkpoint_tracker_filename(model_path)
    iteration, release = read_metadata(tracker_filename)
    head_dim = args.hidden_size // args.num_attention_heads
    group_per_split = args.num_query_groups // args.tensor_model_parallel_size
    if args.num_experts is not None:
        num_local_experts = args.num_experts // args.expert_model_parallel_size
    state_dict = {}
    mid_state = defaultdict(list)
    if (
        args.tensor_model_parallel_size == 1
        and args.pipeline_model_parallel_size == 1
    ):
        checkpoint_name = get_checkpoint_name(model_path, iteration, release, None, None, None, None, None)
        state_dict = torch.load(checkpoint_name)['model']

    elif (
        args.tensor_model_parallel_size > 1
        and args.pipeline_model_parallel_size == 1
        and args.num_experts is None
    ):  
        for tp_rank in range(args.tensor_model_parallel_size):
            checkpoint_name = get_checkpoint_name(model_path, iteration, release, None, tp_rank, None, None, None)
            print(f'load {checkpoint_name}')
            split_state = torch.load(checkpoint_name, map_location="cpu")['model']
            for k, v in split_state.items():
                mid_state[k].append(v)
        for k, v in mid_state.items():
            if not isinstance(v[0], torch.Tensor) or 'norm' in k:
                target_v = v[0]
            elif 'embedding' in k or 'output_layer' in k:
                target_v = torch.cat(v, dim=0)
            elif 'linear_proj' in k or 'linear_fc2' in k:
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
            else:
                raise ValueError
            state_dict[k] = target_v
    elif (
        args.tensor_model_parallel_size > 1
        and args.pipeline_model_parallel_size > 1
        and args.num_experts is None
    ):  
        num_layers = args.num_layers // args.pipeline_model_parallel_size
        layers_to_copy = {}
        for tp_rank in range(args.tensor_model_parallel_size):
            for pp_rank in range(args.pipeline_model_parallel_size):
                layer_offset = pp_rank * num_layers
                for layer in range(num_layers):
                    pp_layer_id = layer + layer_offset
                    layers_to_copy[f"decoder.layers.{layer}"] = pp_layer_id
                checkpoint_name = get_checkpoint_name(model_path, iteration, release, True, tp_rank, pp_rank, None, None)
                print(f'load {checkpoint_name}')
                split_state = torch.load(checkpoint_name, map_location="cpu")['model']
                for k, v in split_state.items():
                    try:
                        pattern = re.compile(r'\d+')
                        res = pattern.findall(k)
                        k = re.sub(r"decoder.layers.\d+", "decoder.layers." + str(layers_to_copy["decoder.layers." + res[0]]), k)
                        mid_state[k].append(v)
                    except:
                        mid_state[k].append(v)
        for k, v in mid_state.items():
            if not isinstance(v[0], torch.Tensor) or 'norm' in k:
                target_v = v[0]
            elif 'embedding' in k or 'output_layer' in k:
                target_v = torch.cat(v, dim=0)
            elif 'linear_proj' in k or 'linear_fc2' in k:
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
            else:
                raise ValueError
            state_dict[k] = target_v

    elif (
        args.tensor_model_parallel_size == 1
        and args.pipeline_model_parallel_size == 1
        and args.expert_model_parallel_size >1
        and args.num_experts % args.expert_model_parallel_size == 0
    ):
        for ep_rank in range(args.expert_model_parallel_size):
            checkpoint_name = get_checkpoint_name(model_path, iteration, release, None, None, None, True, ep_rank)
            print(f'load {checkpoint_name}')
            split_state = torch.load(checkpoint_name, map_location="cpu")['model']
            for k, v in split_state.items():
                if 'local_experts' in k:
                    expert_local_rank = name_to_expert_rank(k)
                    expert_rank = expert_local_rank + num_local_experts * ep_rank
                    k = k.replace(f'local_experts.{expert_local_rank}', f'local_experts.{expert_rank}')
                state_dict[k] = v
    elif (
        args.tensor_model_parallel_size > 1
        and args.pipeline_model_parallel_size == 1
        and args.expert_model_parallel_size > 1
        and args.num_experts % args.expert_model_parallel_size == 0
    ):
        for tp_rank in range(args.tensor_model_parallel_size):
            for ep_rank in range(args.expert_model_parallel_size):
                if args.expert_model_parallel_size >1:
                    checkpoint_name = get_checkpoint_name(model_path, iteration,release, None, tp_rank, None, True, ep_rank)
                elif args.expert_model_parallel_size ==1:
                    checkpoint_name = get_checkpoint_name(model_path, iteration, release, None, tp_rank, None, False)
                print(f'load {checkpoint_name}')
                split_state = torch.load(checkpoint_name, map_location="cpu")['model']
                for k, v in split_state.items():
                    if 'local_experts' in k and 'norm' not in k:
                        local_expert_rank = name_to_expert_rank(k)
                        expert_rank = local_expert_rank + num_local_experts * ep_rank
                        k = k.replace(f'local_experts.{local_expert_rank}', f'local_experts.{expert_rank}')
                        mid_state[k].append(v)
                    elif ep_rank == 0:
                        mid_state[k].append(v)

        for k, v in mid_state.items():
            if not isinstance(v[0], torch.Tensor) or 'norm' in k or 'router' in k or 'gate' in k:
                target_v = v[0]
            elif 'embedding' in k or 'output_layer' in k:
                target_v = torch.cat(v, dim=0)
            elif 'linear_proj' in k or 'linear_fc2' in k:
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
            else:
                print('passed', k)
                exit()
            state_dict[k] = target_v

    elif (
        args.tensor_model_parallel_size > 1
        and args.pipeline_model_parallel_size > 1
        and args.expert_model_parallel_size > 1
        and args.num_experts % args.expert_model_parallel_size == 0
    ):
        num_layers = args.num_layers // args.pipeline_model_parallel_size
        layers_to_copy = {}
        for tp_rank in range(args.tensor_model_parallel_size):
            for ep_rank in range(args.expert_model_parallel_size):
                for pp_rank in range(args.pipeline_model_parallel_size):
                    layer_offset = pp_rank * num_layers
                    for layer in range(num_layers):
                        pp_layer_id = layer + layer_offset
                        layers_to_copy[f"decoder.layers.{layer}"] = pp_layer_id

                    if args.expert_model_parallel_size > 1:
                        checkpoint_name = get_checkpoint_name(model_path, iteration, release, True, tp_rank, pp_rank, True,
                                                              ep_rank)
                    elif args.expert_model_parallel_size == 1:
                        checkpoint_name = get_checkpoint_name(model_path, iteration, release, True, tp_rank, pp_rank,
                                                              False)
                    print(f'load {checkpoint_name}')
                    split_state = torch.load(checkpoint_name, map_location="cpu")['model']
                    for k, v in split_state.items():
                        if 'local_experts' in k and 'norm' not in k:
                            local_expert_rank = name_to_expert_rank(k)
                            expert_rank = local_expert_rank + num_local_experts * ep_rank
                            k = k.replace(f'local_experts.{local_expert_rank}', f'local_experts.{expert_rank}')
                            mid_state[k].append(v)
                        elif ep_rank == 0:
                            mid_state[k].append(v)
                        try:
                            pattern = re.compile(r'\d+')
                            res = pattern.findall(k)
                            k = re.sub(r"decoder.layers.\d+", "decoder.layers." + str(layers_to_copy["decoder.layers." + res[0]]), k)
                            mid_state[k].append(v)
                        except:
                            mid_state[k].append(v)
        for k, v in mid_state.items():
            if not isinstance(v[0], torch.Tensor) or 'norm' in k or 'router' in k or 'gate' in k:
                target_v = v[0]
            elif 'embedding' in k or 'output_layer' in k:
                target_v = torch.cat(v, dim=0)
            elif 'linear_proj' in k or 'linear_fc2' in k:
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
            else:
                raise ValueError
            state_dict[k] = target_v

    else:
        raise ValueError('not support yet')

    model.load_state_dict(state_dict, strict=False)
    return model

"""
    The following two functions convert a TP1PP1 MG/HF model to HF/MG format.
"""

@torch.inference_mode()
def convert_checkpoint_from_megatron_to_transformers(mgmodel, hfmodel, args):
    raise NotImplementedError()
    if args.fp16:
        mgmodel = mgmodel.half()
        hfmodel = hfmodel.half()
    elif args.bf16:
        mgmodel = mgmodel.bfloat16()
        hfmodel = hfmodel.bfloat16()

    num_query_groups = args.num_query_groups
    hidden_size = args.hidden_size
    head_dim = hidden_size // args.num_attention_heads
    use_te = args.transformer_impl == "transformer_engine"
    value_num_per_group = args.num_attention_heads // num_query_groups
    q_dim_per_group = hidden_size // num_query_groups
    kv_dim_per_group = head_dim


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

        qkv_bias = mglayer.self_attention.linear_qkv.bias.view(num_query_groups, -1)
        q_bias, k_bias, v_bias = torch.split(qkv_bias, split_size_or_sections=[q_dim_per_group, kv_dim_per_group, kv_dim_per_group], dim=1)
        q_bias = q_bias.contiguous().view(-1)
        k_bias = k_bias.contiguous().view(-1)
        v_bias = v_bias.contiguous().view(-1)

        hflayer.self_attn.q_proj.bias.copy_(q_bias)
        hflayer.self_attn.k_proj.bias.copy_(k_bias)
        hflayer.self_attn.v_proj.bias.copy_(v_bias)

        hflayer.self_attn.o_proj.weight.copy_(mglayer.self_attention.linear_proj.weight)

        if args.num_experts is None:
            gate_weight, fc1_weight = torch.split(mglayer.mlp.linear_fc1.weight, split_size_or_sections=args.ffn_hidden_size)
            hflayer.mlp.gate_proj.weight.copy_(gate_weight)
            hflayer.mlp.up_proj.weight.copy_(fc1_weight)
            hflayer.mlp.down_proj.weight.copy_(mglayer.mlp.linear_fc2.weight)
        else:
            hflayer.mlp.gate.weight.copy_(mglayer.mlp.router.weight)
            for mgexpert, hfexpert in zip(mglayer.mlp.experts.local_experts, hflayer.mlp.experts):
                gate_weight, up_weight = torch.split(mgexpert.linear_fc1.weight,
                                                        split_size_or_sections=args.moe_ffn_hidden_size)
                hfexpert.gate_proj.weight.copy_(gate_weight)
                hfexpert.up_proj.weight.copy_(up_weight)
                hfexpert.down_proj.weight.copy_(mgexpert.linear_fc2.weight)

            hflayer.mlp.shared_expert_gate.weight.copy_(mglayer.mlp.shared_expert_gate.weight)
            shared_expert_gate_weight, shared_expert_up_weight = \
                torch.split(mglayer.mlp.shared_expert.linear_fc1.weight,
                            split_size_or_sections=args.shared_moe_ffn_hidden_size)
            hflayer.mlp.shared_expert.gate_proj.weight.copy_(shared_expert_gate_weight)
            hflayer.mlp.shared_expert.up_proj.weight.copy_(shared_expert_up_weight)
            hflayer.mlp.shared_expert.down_proj.weight.copy_(mglayer.mlp.shared_expert.linear_fc2.weight)

        if use_te and not args.num_experts:
            hflayer.post_attention_layernorm.weight.copy_(mglayer.mlp.linear_fc1.layer_norm_weight)
        else:
            hflayer.post_attention_layernorm.weight.copy_(mglayer.pre_mlp_layernorm.weight)

    hfmodel.model.norm.weight.copy_(mgmodel.decoder.final_layernorm.weight)
    if args.untie_embeddings_and_output_weights:
        hfmodel.lm_head.weight.copy_(mgmodel.output_layer.weight)

def safe_copy(src_tensor: torch.Tensor, dst_tensor: torch.Tensor):
    assert src_tensor.dtype == dst_tensor.dtype
    assert src_tensor.shape == dst_tensor.shape
    dst_tensor.data.copy_(src_tensor.data)
    return src_tensor.numel()

@torch.inference_mode()
def convert_checkpoint_from_transformers_to_megatron(hfmodel, mgmodel, args):
    if args.fp16:
        mgmodel = mgmodel.half()
        hfmodel = hfmodel.half()
    elif args.bf16:
        mgmodel = mgmodel.bfloat16()
        hfmodel = hfmodel.bfloat16()

    # assert args.num_query_groups >= args.target_tensor_model_parallel_size

    num_attention_heads = args.num_attention_heads
    num_query_groups = args.num_query_groups
    hidden_size = args.hidden_size
    head_dim = hidden_size // num_attention_heads

    # 1. vision model
    hfvision = hfmodel.visual
    mgvision = mgmodel.vision_model
    vision_hidden_size = mgvision.config.hidden_size
    vision_num_query_groups = mgvision.config.num_query_groups
    vision_head_dim = vision_hidden_size // mgvision.config.num_attention_heads
    copied_numel = 0
    safe_copy(hfvision.rotary_pos_emb.inv_freq, mgvision.rotary_pos_emb.inv_freq)
    copied_numel += safe_copy(hfvision.patch_embed.proj.weight, mgvision.patch_embed.proj.weight)
    for hfblock, mgblock in zip(hfvision.blocks, mgvision.decoder.layers):
        # norm1 --> linear_qkv.norm
        copied_numel += safe_copy(hfblock.norm1.weight, mgblock.self_attention.linear_qkv.layer_norm_weight)
        copied_numel += safe_copy(hfblock.norm1.bias, mgblock.self_attention.linear_qkv.layer_norm_bias)
        # norm2 --> mlp.linear_fc1.norm
        copied_numel += safe_copy(hfblock.norm2.weight, mgblock.mlp.linear_fc1.layer_norm_weight)
        copied_numel += safe_copy(hfblock.norm2.bias, mgblock.mlp.linear_fc1.layer_norm_bias)       
        # qkv --> self_attention.linear_qkv
        converted_weight = (
            hfblock.attn.qkv.weight
            .view(3, vision_num_query_groups, -1, vision_head_dim, vision_hidden_size)
            .transpose(0, 1)
            .flatten(1, 2)
            .reshape(-1, vision_hidden_size)
            .contiguous()
        )
        copied_numel += safe_copy(converted_weight, mgblock.self_attention.linear_qkv.weight)
        converted_bias = (
            hfblock.attn.qkv.bias
            .view(3, vision_num_query_groups, -1)
            .transpose(0, 1)
            .flatten(1, 2)
            .view(-1)
            .contiguous()
        )
        copied_numel += safe_copy(converted_bias, mgblock.self_attention.linear_qkv.bias)
        # proj --> self_attention.linear_proj
        copied_numel += safe_copy(hfblock.attn.proj.weight, mgblock.self_attention.linear_proj.weight)
        copied_numel += safe_copy(hfblock.attn.proj.bias, mgblock.self_attention.linear_proj.bias)
        # mlp --> mlp: no gate
        copied_numel += safe_copy(hfblock.mlp.fc1.weight, mgblock.mlp.linear_fc1.weight)
        copied_numel += safe_copy(hfblock.mlp.fc1.bias, mgblock.mlp.linear_fc1.bias)
        copied_numel += safe_copy(hfblock.mlp.fc2.weight, mgblock.mlp.linear_fc2.weight)
        copied_numel += safe_copy(hfblock.mlp.fc2.bias, mgblock.mlp.linear_fc2.bias)        

    # 2. vision projector
    hfprojector = hfvision.merger
    mgprojector = mgvision.projection
    copied_numel += safe_copy(hfprojector.ln_q.weight, mgvision.decoder.final_layernorm.weight)
    copied_numel += safe_copy(hfprojector.ln_q.bias, mgvision.decoder.final_layernorm.bias)   

    copied_numel += safe_copy(hfprojector.mlp[0].weight, mgprojector.encoder.linear_fc1.weight)
    copied_numel += safe_copy(hfprojector.mlp[0].bias, mgprojector.encoder.linear_fc1.bias)
    copied_numel += safe_copy(hfprojector.mlp[2].weight, mgprojector.encoder.linear_fc2.weight)
    copied_numel += safe_copy(hfprojector.mlp[2].bias, mgprojector.encoder.linear_fc2.bias)
    n_params = sum([t.numel() for t in hfvision.state_dict().values()])
    assert n_params == copied_numel

    # 3. llm [just Qwen2]
    hfllm = hfmodel.model
    mgllm = mgmodel.language_model
    copied_numel = 0
    copied_numel += safe_copy(hfllm.embed_tokens.weight, mgllm.embedding.word_embeddings.weight)
    for mglayer, hflayer in zip(mgllm.decoder.layers, hfllm.layers):
        copied_numel += safe_copy(hflayer.input_layernorm.weight, mglayer.self_attention.linear_qkv.layer_norm_weight)
        
        q_proj_weight = hflayer.self_attn.q_proj.weight.view(num_query_groups, -1, head_dim, hidden_size)
        k_proj_weight = hflayer.self_attn.k_proj.weight.view(num_query_groups, -1, head_dim, hidden_size)
        v_proj_weight = hflayer.self_attn.v_proj.weight.view(num_query_groups, -1, head_dim, hidden_size)
        qkv_proj = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=1).view(-1, hidden_size).contiguous()
        copied_numel += safe_copy(qkv_proj, mglayer.self_attention.linear_qkv.weight)

        q_proj_bias = hflayer.self_attn.q_proj.bias.view(num_query_groups, -1)
        k_proj_bias = hflayer.self_attn.k_proj.bias.view(num_query_groups, -1)
        v_proj_bias = hflayer.self_attn.v_proj.bias.view(num_query_groups, -1)
        qkv_bias = torch.cat([q_proj_bias, k_proj_bias, v_proj_bias], dim=1).view(-1).contiguous()
        copied_numel += safe_copy(qkv_bias, mglayer.self_attention.linear_qkv.bias)
        copied_numel += safe_copy(hflayer.self_attn.o_proj.weight, mglayer.self_attention.linear_proj.weight)

        fc1_weight = torch.cat([hflayer.mlp.gate_proj.weight, hflayer.mlp.up_proj.weight])
        copied_numel += safe_copy(fc1_weight, mglayer.mlp.linear_fc1.weight)

        copied_numel += safe_copy(hflayer.mlp.down_proj.weight, mglayer.mlp.linear_fc2.weight)
        copied_numel += safe_copy(hflayer.post_attention_layernorm.weight, mglayer.mlp.linear_fc1.layer_norm_weight)

    copied_numel += safe_copy(hfllm.norm.weight, mgllm.decoder.final_layernorm.weight)
    if args.untie_embeddings_and_output_weights:
        safe_copy(hfmodel.lm_head.weight, mgllm.output_layer.weight)
    
    n_params = sum([t.numel() for t in hfllm.state_dict().values()])
    assert n_params == copied_numel

def save_state_dict(args, model, checkpoint_name):
    state_dict = {}
    state_dict['args'] = args
    state_dict['checkpoint_version'] = 3.0
    state_dict['iteration'] = 0    
    state_dict['model'] = model
    os.makedirs(os.path.dirname(checkpoint_name), exist_ok=True)
    print(f'save model part {checkpoint_name}')
    torch.save(clone_state_dict(state_dict), checkpoint_name)

def check_layer(layers_to_copy, k):
    pattern = re.compile(r"decoder.layers.\d+")
    res = pattern.findall(k)
    return res and res[0] in layers_to_copy.keys()

def split_vision_model(mgvision, args, prefix="vision_model") -> Dict[Tuple, Dict]:
    state_dicts = {}
    tp = args.tensor_model_parallel_size
    ENCODER_NUM_ATTENTION_HEADS = 16

    full_model = mgvision.state_dict_for_save_checkpoint()
    num_query_groups = mgvision.config.num_query_groups
    group_per_split = num_query_groups // tp

    vision_hidden_size = mgvision.config.hidden_size
    vision_ffn_hidden_size = mgvision.config.ffn_hidden_size
    head_dim = vision_hidden_size // ENCODER_NUM_ATTENTION_HEADS
    assert ENCODER_NUM_ATTENTION_HEADS % tp == 0
    assert tp > 1, "TP1 should not call this function!"
    # split model with ETP
    for etp_rank in range(tp):
        d = {}
        for k, v in full_model.items():
            if not isinstance(v, torch.Tensor):
                target_v = v
            elif 'patch_embed' in k:
                target_v = v
            elif 'linear_qkv.weight' in k:
                viewed = v.view(num_query_groups, -1, head_dim, vision_hidden_size)
                viewed = viewed[group_per_split*etp_rank : group_per_split*(etp_rank + 1)]
                target_v = viewed.view(-1, vision_hidden_size)
            elif 'linear_qkv.bias' in k:
                viewed = v.view(num_query_groups, -1, head_dim)
                viewed = viewed[group_per_split * etp_rank: group_per_split * (etp_rank + 1)]
                target_v = viewed.view(-1)
            elif ('linear_proj' in k or 'linear_fc2' in k) and 'bias' not in k:
                seg = v.shape[-1] // tp
                target_v = v[..., seg*etp_rank : seg*(etp_rank + 1)]
            elif 'linear_fc1.weight' in k and 'projection' not in k:
                viewed = v.view(-1, vision_ffn_hidden_size, vision_hidden_size)
                seg = vision_ffn_hidden_size // tp
                target_v = viewed[:, seg*etp_rank: seg*(etp_rank+1), :].reshape(-1, vision_hidden_size)
            elif 'linear_fc1.weight' in k:
                viewed = v.view(-1, vision_ffn_hidden_size, vision_ffn_hidden_size)
                seg = vision_ffn_hidden_size // tp
                target_v = viewed[:, seg*etp_rank: seg*(etp_rank+1), :].reshape(-1, vision_ffn_hidden_size)                    
            elif 'linear_fc1.bias' in k:
                viewed = v.view(-1, vision_ffn_hidden_size)
                seg = vision_ffn_hidden_size // tp
                target_v = viewed[:, seg*etp_rank: seg*(etp_rank+1)].flatten()
            else:
                target_v = v                
            d[prefix + '.' + k] = target_v
        state_dicts[(etp_rank, 0)] = d
    return state_dicts

def save_mgmodel(mgmodel, args):
    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size

    os.makedirs(args.save, exist_ok=True)
    os.system("cp -rf " + args.load + "/config*.json " + args.save)
    os.system("cp -rf " + args.load + "/tokenizer* " + args.save)
    os.system("cp -rf " + args.load + "/vocab.json " + args.save)
    os.system("cp -rf " + args.load + "/merges.txt " + args.save)

    tracker_filepath = os.path.join(args.save, 'latest_checkpointed_iteration.txt')
    with open(tracker_filepath, "w") as f:
        f.write("release")

    head_dim = args.hidden_size // args.num_attention_heads
    group_per_split = args.num_query_groups // args.target_tensor_model_parallel_size
    full_model = mgmodel.state_dict_for_save_checkpoint()
    num_layers = args.num_layers // args.pipeline_model_parallel_size
    for k in list(full_model.keys()):
        if full_model[k] is None:
            full_model.pop(k)

    if (
        args.tensor_model_parallel_size == 1
        and args.pipeline_model_parallel_size == 1
    ):
        checkpoint_name = get_checkpoint_name(args.save, 0, True)
        save_state_dict(args, full_model, checkpoint_name)
    elif (
        args.tensor_model_parallel_size > 1
        and args.pipeline_model_parallel_size == 1
    ):
        vision_state_dicts = split_vision_model(mgmodel.vision_model, args)
        for tp_rank in range(args.tensor_model_parallel_size):
            model_split = {}
            checkpoint_name = get_checkpoint_name(args.save, 0, True, None, tp_rank)
            print(f'tensor_parallel, save model to {checkpoint_name}')
            for k, v in full_model.items():
                if not isinstance(v, torch.Tensor):
                    target_v = v
                elif 'vision_model' in k:
                    vision_part = vision_state_dicts[(tp_rank, 0)]
                    assert k in vision_part, f"Cannot find key {k} in vision model split!"
                    target_v = vision_part[k]
                elif 'linear_qkv.weight' in k:
                    viewed = v.view(args.num_query_groups, -1, head_dim, args.hidden_size)
                    viewed = viewed[group_per_split*tp_rank : group_per_split*(tp_rank + 1)]
                    target_v = viewed.view(-1, args.hidden_size)
                elif 'linear_qkv.bias' in k:
                    viewed = v.view(args.num_query_groups, -1, head_dim)
                    viewed = viewed[group_per_split * tp_rank: group_per_split * (tp_rank + 1)]
                    target_v = viewed.view(-1)
                elif 'linear_proj' in k or 'linear_fc2' in k:
                    seg = v.shape[1] // args.tensor_model_parallel_size
                    target_v = v[:, seg*tp_rank : seg*(tp_rank + 1)]
                elif 'embedding' in k or 'output_layer' in k:
                    seg = v.shape[0] // args.tensor_model_parallel_size
                    target_v = v[seg*tp_rank : seg*(tp_rank + 1)]
                elif 'linear_fc1' in k and 'norm' not in k:
                    viewed = v.view(-1, args.ffn_hidden_size, args.hidden_size)
                    seg = args.ffn_hidden_size // args.tensor_model_parallel_size
                    target_v = viewed[:, seg*tp_rank: seg*(tp_rank+1), :].reshape(-1, args.hidden_size)
                else:
                    target_v = v
                model_split[k] = target_v
            save_state_dict(args, model_split, checkpoint_name)
    elif (
        args.pipeline_model_parallel_size > 1
    ):
        vision_state_dicts = split_vision_model(mgmodel.vision_model, args)
        for tp_rank in range(args.tensor_model_parallel_size):
            for pp_rank in range(args.pipeline_model_parallel_size):
                model_split = {}
                layer_offset = pp_rank * num_layers
                layers_to_copy = {}
                for layer in range(num_layers):
                    pp_layer_id = layer + layer_offset
                    layers_to_copy[f"decoder.layers.{pp_layer_id}"] = layer
                checkpoint_name = get_checkpoint_name(args.save, 0, True, True, tp_rank, pp_rank)
                print(f'tensor_parallel & pipeline_parallel, save model to {checkpoint_name}')
                for k, v in full_model.items():
                    if check_layer(layers_to_copy, k) and 'vision_model' not in k:
                        pattern = re.compile(r'\d+')
                        res = pattern.findall(k)
                        k = re.sub(r"decoder.layers.\d+", "decoder.layers." + str(layers_to_copy["decoder.layers." + res[0]]), k)
                    elif not ("word_embeddings" in k or "output_layer" in k or "final_layernorm" in k or 'vision_model' in k):
                        continue
                    if 'vision_model' in k:
                        if pp_rank > 0:
                            continue
                        vision_part = vision_state_dicts[(tp_rank, 0)]
                        assert k in vision_part, f"Cannot find key {k} in vision model split!"
                        target_v = vision_part[k]
                    elif not isinstance(v, torch.Tensor):
                        target_v = v
                    elif 'linear_qkv.weight' in k:
                        viewed = v.view(args.num_query_groups, -1, head_dim, args.hidden_size)
                        viewed = viewed[group_per_split*tp_rank : group_per_split*(tp_rank + 1)]
                        target_v = viewed.view(-1, args.hidden_size)
                    elif 'linear_qkv.bias' in k:
                        viewed = v.view(args.num_query_groups, -1, head_dim)
                        viewed = viewed[group_per_split * tp_rank: group_per_split * (tp_rank + 1)]
                        target_v = viewed.view(-1)
                    elif 'linear_proj' in k or 'linear_fc2' in k:
                        seg = v.shape[1] // args.tensor_model_parallel_size
                        target_v = v[:, seg*tp_rank : seg*(tp_rank + 1)]
                    elif 'embedding' in k or 'output_layer' in k:
                        seg = v.shape[0] // args.tensor_model_parallel_size
                        target_v = v[seg*tp_rank : seg*(tp_rank + 1)]
                    elif 'linear_fc1' in k and 'norm' not in k:
                        viewed = v.view(-1, args.ffn_hidden_size, args.hidden_size)
                        seg = args.ffn_hidden_size // args.tensor_model_parallel_size
                        target_v = viewed[:, seg*tp_rank: seg*(tp_rank+1), :].reshape(-1, args.hidden_size)
                    else:
                        target_v = v
                    if "word_embeddings" in k:
                        if pp_rank == 0:
                            model_split[k] = target_v
                    elif 'vision_model' not in k and ("output_layer" in k or "final_layernorm" in k):
                        if pp_rank == args.pipeline_model_parallel_size - 1:
                            model_split[k] = target_v
                    else:
                        model_split[k] = target_v
                save_state_dict(args, model_split, checkpoint_name)

    else:
        raise ValueError('Something is wrong, please check your tp/pp size')

    print(f'megatron model is save to {args.save}')


def save_hfmodel(args, model):
    output_state_dict = model.state_dict()
    max_shard_size = "10GB"
    # TODO: fix safetensor save
    shards, index = shard_checkpoint(output_state_dict, max_shard_size=max_shard_size)
    os.makedirs(args.save, exist_ok=True)
    for shard_file, shard in shards.items():
        if args.save_safetensors:
            shard_file = shard_file.replace("pytorch_", "")
            shard_file = shard_file.replace(".bin", ".safetensors")
            target_file = os.path.join(args.save, shard_file)
            print(f'huggingface model is save to {target_file}')
            new_shard = {}
            for k, v in shard.items():
                new_shard[k] = copy.deepcopy(v)
            safetensors.torch.save_file(clone_state_dict(new_shard), target_file, metadata={"format": "pt"})
        else:
            target_file = os.path.join(args.save, shard_file)
            print(f'huggingface model is save to {target_file}')
            torch.save(clone_state_dict(shard), target_file)

    if index is None:
        print(f"Model weights saved in {os.path.join(args.save, WEIGHTS_NAME)}")
    else:
        save_index_file = os.path.join(args.save, WEIGHTS_INDEX_NAME)
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
        print(
            f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
            f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
            f"index located at {save_index_file}."
        )

def check_hf_mg_forward(hfmodel, mgmodel, mgargs):
    print(hfmodel)
    print(mgmodel)

    hf_hiddens = [{} for _ in range(mgargs.num_layers)]
    mg_hiddens = [{} for _ in range(mgargs.num_layers)]

    vision_hidden_size = mgmodel.vision_model.config.hidden_size
    hidden_size = mgargs.hidden_size
    vocab_size = mgargs.padded_vocab_size
    vision_ffn_hidden_size = mgmodel.vision_model.config.ffn_hidden_size
    def print_input_hook(module, args, kwargs, layer_idx, mode):
        frame, name, *others = mode.split('-')
        if len(others) > 0:
            n = '-'.join([name, *others])
            if frame == 'hf':
                hf_hiddens[layer_idx][n] = args[0][:, None]
            elif frame == 'mg' and 'layer' in mode:
                mg_hiddens[layer_idx][n] = kwargs.get('hidden_states')
            elif frame == 'mg':
                mg_hiddens[layer_idx][n] = args[0]
        elif frame == 'hf':
            hf_hiddens[layer_idx][name] = args[0].transpose(0, 1)
        elif frame == 'mg' and 'layer' in mode:
            mg_hiddens[layer_idx][name] = kwargs.get('hidden_states')
        elif frame == 'mg':
            mg_hiddens[layer_idx][name] = args[0]

    def print_output_hook(module, args, kwargs, output, layer_idx, mode):
        frame, name, *others = mode.split('-')
        if mode in ['hf-lmhead']:
            hf_hiddens[layer_idx][name] = output.transpose(0, 1).reshape(-1, vocab_size)
            hf_hiddens[layer_idx][name + "_weight"] = module.weight
            hf_hiddens[layer_idx][name + '_token'] = output.transpose(0, 1).max(dim=-1)[1]
        elif mode in ['mg-lmhead']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, vocab_size)
            mg_hiddens[layer_idx][name + "_weight"] = module.weight
            mg_hiddens[layer_idx][name + '_token'] = output[0].max(dim=-1)[1]
        elif mode in ['hf-o_proj_out']:
            hf_hiddens[layer_idx][name] = output
            hf_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['mg-o_proj_out']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, hidden_size)
            mg_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['hf-attn_out']:
            hf_hiddens[layer_idx][name] = output[0].reshape(-1, hidden_size)
        elif mode in ['mg-attn_out']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, hidden_size)
        elif mode in ['hf-patchembed']:
            hf_hiddens[layer_idx][name] = output
        elif mode in ['mg-patchembed']:
            mg_hiddens[layer_idx][name] = output
        elif len(others) > 0:
            n = '-'.join([name, *others])
            if name in ['o_proj_out', 'mlp_out']:
                if frame == 'hf':
                    hf_hiddens[layer_idx][n] = output
                else:
                    if isinstance(output, (list, tuple)):
                        output = output[0]
                    mg_hiddens[layer_idx][n] = output.reshape(-1, vision_hidden_size)
            elif name == 'attn_out':
                if frame == 'hf':
                    hf_hiddens[layer_idx][n] = output.reshape(-1, vision_hidden_size)
                else:
                    mg_hiddens[layer_idx][n] = (output[0] + output[1]).reshape(-1, vision_hidden_size)
            elif name == 'fc1_out':
                if frame == 'hf':
                    hf_hiddens[layer_idx][n] = output
                else:
                    if isinstance(output, (list, tuple)):
                        output = output[0]
                    mg_hiddens[layer_idx][n] = output.reshape(-1, vision_ffn_hidden_size)
    # vision hook: patch_embed, decoder, projection
    hfmodel.visual.patch_embed.register_forward_hook(
        partial(print_output_hook, layer_idx=0, mode='hf-patchembed'), with_kwargs=True)

    mgmodel.vision_model.patch_embed.register_forward_hook(
        partial(print_output_hook, layer_idx=0, mode='mg-patchembed'), with_kwargs=True)

    for idx, layer in enumerate(hfmodel.visual.blocks):
        layer.register_forward_pre_hook(partial(print_input_hook, layer_idx=0, mode=f'hf-layer_in-vision{idx}'), with_kwargs=True)

        layer.attn.proj.register_forward_pre_hook(partial(print_input_hook, layer_idx=0, mode=f'hf-o_proj_in-vision{idx}'),
                                                         with_kwargs=True)

        layer.attn.proj.register_forward_hook(partial(print_output_hook, layer_idx=0, mode=f'hf-o_proj_out-vision{idx}'),
                                                     with_kwargs=True)

        layer.attn.register_forward_hook(partial(print_output_hook, layer_idx=0, mode=f'hf-attn_out-vision{idx}'),
                                              with_kwargs=True)
        
        layer.mlp.fc1.register_forward_hook(partial(print_output_hook, layer_idx=0, mode=f'hf-fc1_out-vision{idx}'),
                                              with_kwargs=True)
        
        layer.mlp.register_forward_hook(partial(print_output_hook, layer_idx=0, mode=f'hf-mlp_out-vision{idx}'),
                                              with_kwargs=True)



    for idx, layer in enumerate(mgmodel.vision_model.decoder.layers):
        layer.register_forward_pre_hook(partial(print_input_hook, layer_idx=0, mode=f'mg-layer_in-vision{idx}'), with_kwargs=True)

        layer.self_attention.linear_proj.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=0, mode=f'mg-o_proj_in-vision{idx}'), with_kwargs=True)

        layer.self_attention.linear_proj.register_forward_hook(
            partial(print_output_hook, layer_idx=0, mode=f'mg-o_proj_out-vision{idx}'), with_kwargs=True)

        layer.self_attention.register_forward_hook(partial(print_output_hook, layer_idx=0, mode=f'mg-attn_out-vision{idx}'),
                                                   with_kwargs=True)
            
        layer.mlp.linear_fc1.register_forward_hook(partial(print_output_hook, layer_idx=0, mode=f'mg-fc1_out-vision{idx}'),
                                                   with_kwargs=True)

        layer.mlp.register_forward_hook(partial(print_output_hook, layer_idx=0, mode=f'mg-mlp_out-vision{idx}'),
                                              with_kwargs=True)



    hfmodel.visual.merger.ln_q.register_forward_hook(
        partial(print_output_hook, layer_idx=0, mode='hf-finalln'), with_kwargs=True)

    mgmodel.vision_model.decoder.final_layernorm.register_forward_hook(
        partial(print_output_hook, layer_idx=0, mode='mg-finalln'), with_kwargs=True)

    if mgargs.untie_embeddings_and_output_weights:
        hfmodel.lm_head.register_forward_hook(partial(print_output_hook, layer_idx=mgargs.num_layers - 1, mode='hf-lmhead'),
                                            with_kwargs=True)

        mgmodel.language_model.output_layer.register_forward_hook(
            partial(print_output_hook, layer_idx=mgargs.num_layers - 1, mode='mg-lmhead'), with_kwargs=True)

    for idx, layer in enumerate(hfmodel.model.layers):

        layer.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='hf-layer_in'), with_kwargs=True)

        layer.self_attn.o_proj.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='hf-o_proj_in'),
                                                         with_kwargs=True)

        layer.self_attn.o_proj.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='hf-o_proj_out'),
                                                     with_kwargs=True)

        layer.self_attn.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='hf-attn_out'),
                                              with_kwargs=True)


    for idx, layer in enumerate(mgmodel.language_model.decoder.layers):

        layer.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='mg-layer_in'), with_kwargs=True)

        layer.self_attention.linear_proj.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode='mg-o_proj_in'), with_kwargs=True)

        layer.self_attention.linear_proj.register_forward_hook(
            partial(print_output_hook, layer_idx=idx, mode='mg-o_proj_out'), with_kwargs=True)

        layer.self_attention.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='mg-attn_out'),
                                                   with_kwargs=True)

    # prepare visual inputs
    pixel_values = torch.randn(512, 1176).cuda()
    image_grid_thw = torch.tensor([[1, 16, 32]]).cuda()

    input_ids = torch.tensor([[151644,   8506,  22564,  27608,  75188,   4344, 121395,  61991,  79554, 36689]]).long().cuda()
    position_ids, _ = hfmodel.get_rope_index(input_ids)

    is_oom = False
    with torch.inference_mode():
        try:
            hfmodel.cuda()
            hf_vision_output = hfmodel.visual(pixel_values, grid_thw=image_grid_thw)
            hf_inputs_embeds = hfmodel.model.embed_tokens(input_ids)
            hflogits = hfmodel.lm_head(hfmodel.model(
                input_ids=None,
                inputs_embeds=hf_inputs_embeds,
                position_ids=position_ids
            )[0])
        except torch.cuda.OutOfMemoryError:
            print('oom for huggingface model forward')
            is_oom = True
        hfmodel.cpu()
        del hfmodel

    with torch.inference_mode():
        try:
            mgmodel.cuda()
            mg_vision_output = mgmodel.vision_model(
                vision_data=pixel_values,
                grid_thw=image_grid_thw
            )
            language_embeddings = mgmodel.language_model.embedding(
                input_ids=input_ids,
                position_ids=None # NOTE: disable
            ).clone()
            mglogits = mgmodel.language_model(
                input_ids=None,
                position_ids=position_ids,    
                decoder_input=language_embeddings, 
                attention_mask=None
            )
        except torch.cuda.OutOfMemoryError:
            print('oom for megatron model forward')
            is_oom = True
        mgmodel.cpu()
        del mgmodel

    epsilon = 1e-5
    for idx, (hfh, mgh) in enumerate(zip(hf_hiddens, mg_hiddens)):
        assert len(hfh) == len(mgh)
        for k, hfv in hfh.items():
            mgv, hfv = mgh[k].cpu(), hfv.cpu()
            same_num = (hfv != mgv).sum()
            diff_num = ((hfv - mgv) > epsilon).sum()
            diff_max = (hfv - mgv).abs().max()
            print(f'layer:{idx}, {k}, diff: {same_num}, diff>{epsilon}:[{diff_num}/{hfv.numel()}] diff_max:{diff_max}')

    if not is_oom:
        vis_diff_num = ((hf_vision_output - mg_vision_output) > epsilon).sum()
        vis_diff_max = ((hf_vision_output - mg_vision_output)).abs().max()
        diff_num = ((hflogits - mglogits) > epsilon).sum()
        diff_max = (hflogits - mglogits).abs().max()
        print(f'visual: diff>{epsilon}:[{vis_diff_num}/{hf_vision_output.numel()}] diff_max:{vis_diff_max}')
        print(f'logits: diff>{epsilon}:[{diff_num}/{hflogits.numel()}] diff_max:{diff_max}')

def add_extra_args(parser):
    parser = get_patch_args(parser)
    parser = add_model_args(parser)
    return parser

def main():
    initialize_megatron(extra_args_provider=add_extra_args)
    args = get_args()

    if False and args.convert_checkpoint_from_megatron_to_transformers:
        config = AutoConfig.from_pretrained(args.hf_ckpt_path)
        hf_model = Qwen2VLForConditionalGeneration.from_pretrained(args.hf_ckpt_path, torch_dtype=config.torch_dtype)
        mg_model = load_megatron_model(args)
        convert_checkpoint_from_megatron_to_transformers(mg_model, hf_model, args)
        save_hfmodel(args, hf_model)
    else:
        config = AutoConfig.from_pretrained(args.load)
        hf_model = Qwen2VLForConditionalGeneration.from_pretrained(args.load, torch_dtype=config.torch_dtype)
        mg_model = model_provider()
        convert_checkpoint_from_transformers_to_megatron(hf_model, mg_model, args)
        # check_hf_mg_forward(hf_model, mg_model, args)
        save_mgmodel(mg_model, args)

if __name__ == "__main__":
    main()

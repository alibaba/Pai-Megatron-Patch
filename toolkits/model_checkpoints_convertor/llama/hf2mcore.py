import os
import re
import json
import torch
import transformers
import torch.nn as nn
from functools import partial
from collections import defaultdict
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, shard_checkpoint, load_sharded_checkpoint
from megatron.initialize import initialize_megatron
from megatron import get_args
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, read_metadata

import sys

path_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.append(os.path.join(path_dir, "examples"))
from llama2.pretrain_mcore_llama import model_provider
from megatron_patch.arguments import get_patch_args

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

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

def add_checkpointing_args(parser):
    parser.add_argument('--megatron-path',
                        type=str,
                        default=None,
                        help='Base directory of Megatron repository')

    parser.add_argument(
        '--convert_checkpoint_from_megatron_to_transformers',
        action='store_true',
        help=
        ('If True, convert a Megatron checkpoint to a Transformers checkpoint. '
         'If False, convert a Transformers checkpoint to a Megatron checkpoint.'
         ),
    )

    parser.add_argument(
        '--load_path',
        type=str,
        required=True,
        help='Path to the checkpoint to convert.',
    )

    parser.add_argument(
        '--save_path',
        type=str,
        required=True,
        help='Path to the converted checkpoint.',
    )

    parser.add_argument(
        '--huggingface_model_path',
        type=str,
        required=
        True,
    )

    return parser


def add_megatron_checkpoint_args(parser):
    parser.add_argument(
        "--target_tensor_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The tensor model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )

    parser.add_argument(
        "--target_pipeline_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The pipeline model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )

    parser.add_argument(
        "--target_expert_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The data parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )

    parser.add_argument(
        "--num_expert_split_size",
        type=int,
        default=1
    )

    return parser


def add_transformers_checkpoint_args(parser):
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="10GB",
        help=(
            "The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size "
            "lower than this size. If expressed as a string, needs to be digits followed by a unit (like `5MB`). "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )

    return parser


def build_huggingface_model(model_to_load, compute_dtype, random_init=False):
    config = AutoConfig.from_pretrained(
        model_to_load,
        trust_remote_code=True,
    )

    if random_init:
        model = AutoModelForCausalLM.from_config(
            config=config,
            torch_dtype=compute_dtype,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_to_load,
            torch_dtype=compute_dtype,
            trust_remote_code=True
        )

    return config, model.eval()


def replace_mlp_with_moe(args, model):
    config = MixtralConfig(
        intermediate_size=args.intermediate_size,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_local_experts=args.num_local_experts,
        num_key_value_heads=args.num_key_value_heads,
        rope_theta=args.rope_theta,
        rms_norm_eps=args.rms_norm_eps,
        num_experts_per_tok=1,
    )

    def get_hidden_output(module, args, output):
        return output[0]

    for layer in model.model.layers:
        mlp = MixtralSparseMoeBlock(config).to(args.torch_dtype)
        mlp.register_forward_hook(get_hidden_output)
        layer.mlp = mlp

    return model


def create_huggingface_model(args):
    if not args.convert_checkpoint_from_megatron_to_transformers or args.num_experts is None:
        copy_huggingface_tokenizer(args.huggingface_model_path, args.save_path)
        config, model = build_huggingface_model(args.huggingface_model_path, args.params_dtype)
    else:
        copy_huggingface_tokenizer(args.huggingface_model_path, args.save_path, with_code=True)
        config, model = build_huggingface_model(args.save_path, args.params_dtype, random_init=True)
        model = replace_mlp_with_moe(config, model)

    return config, model.eval()


def create_megatron_model(args, hf_config):
    args.hidden_size = hf_config.hidden_size
    args.num_layers = hf_config.num_hidden_layers
    args.num_attention_heads = hf_config.num_attention_heads
    args.kv_channels = args.hidden_size // args.num_attention_heads
    if not args.convert_checkpoint_from_megatron_to_transformers:
        if args.num_expert_split_size == 1:
            args.ffn_hidden_size = hf_config.intermediate_size
        else:
            args.ffn_hidden_size = hf_config.intermediate_size // args.num_expert_split_size
    else:
        args.ffn_hidden_size = hf_config.intermediate_size
    args.num_query_groups = hf_config.num_key_value_heads
    model = model_provider()
    return model.eval()


def copy_huggingface_tokenizer(src_path, dst_path, with_code=False):
    assert os.path.exists(src_path)
    os.makedirs(dst_path, exist_ok=True)
    os.system("cp -rf " + src_path + "/config*.json " + dst_path)
    os.system("cp -rf " + src_path + "/tokenizer* " + dst_path)
    if with_code:
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        code_path = os.path.join(cur_dir, 'hf_llama_moe')
        os.system("cp -rf " + code_path + "/*.py " + dst_path)
        os.system("cp -rf " + code_path + "/*.json " + dst_path)


def name_to_expert_rank(key):
    pattern = r'local_experts\.(\d+)\.'
    expert_rank = int(re.findall(pattern, key)[0])
    return expert_rank


def load_megatron_model(args, model):
    model_path = args.load_path
    tracker_filename = get_checkpoint_tracker_filename(model_path)
    iteration, release = read_metadata(tracker_filename)
    head_dim = args.hidden_size // args.num_attention_heads
    group_per_split = args.num_query_groups // args.target_tensor_model_parallel_size
    num_local_experts = args.num_experts // args.target_expert_model_parallel_size if args.num_experts else 0
    state_dict = {}
    mid_state = defaultdict(list)
    if (
            args.target_tensor_model_parallel_size == 1
            and args.target_pipeline_model_parallel_size == 1
            and args.target_expert_model_parallel_size == 1
    ):
        checkpoint_name = get_checkpoint_name(model_path, iteration, release, None, None, None, None, None)
        state_dict = torch.load(checkpoint_name)['model']
    elif (
            args.target_tensor_model_parallel_size == 1
            and args.target_pipeline_model_parallel_size == 1
            and args.num_experts
            and args.num_experts % args.target_expert_model_parallel_size == 0
    ):
        for ep_rank in range(args.target_expert_model_parallel_size):
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
            args.target_tensor_model_parallel_size > 1
            and args.target_pipeline_model_parallel_size == 1
            and args.num_experts is None
    ):
        for tp_rank in range(args.target_tensor_model_parallel_size):
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
            args.target_tensor_model_parallel_size > 1
            and args.target_pipeline_model_parallel_size == 1
            and args.num_experts
            and args.num_experts % args.target_expert_model_parallel_size == 0
    ):
        for tp_rank in range(args.target_tensor_model_parallel_size):
            for ep_rank in range(args.target_expert_model_parallel_size):
                checkpoint_name = get_checkpoint_name(model_path, iteration, release, None, tp_rank, None, True,
                                                      ep_rank)
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
            if not isinstance(v[0], torch.Tensor) or 'norm' in k or 'router' in k:
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
            state_dict[k] = target_v
    else:
        raise ValueError('not support yet')

    model.load_state_dict(state_dict)
    return model


def convert_checkpoint_from_megatron_to_transformers(mgmodel, hgmodel, args):
    query_group = args.num_query_groups
    hidden_size = args.hidden_size
    head_dim = hidden_size // args.num_attention_heads
    num_experts = args.num_experts
    value_num_per_group = args.num_attention_heads // query_group
    with torch.no_grad():
        hgmodel.model.embed_tokens.weight.copy_(mgmodel.embedding.word_embeddings.weight)
        for mglayer, hglayer in zip(mgmodel.decoder.layers, hgmodel.model.layers):
            hglayer.input_layernorm.weight.copy_(mglayer.self_attention.linear_qkv.layer_norm_weight)
            qkv_weight = mglayer.self_attention.linear_qkv.weight.view(query_group, -1, head_dim, hidden_size)
            q_weight, k_weight, v_weight = torch.split(qkv_weight, split_size_or_sections=[value_num_per_group, 1, 1],
                                                       dim=1)
            hglayer.self_attn.q_proj.weight.copy_(q_weight.reshape(-1, hidden_size))
            hglayer.self_attn.k_proj.weight.copy_(k_weight.reshape(-1, hidden_size))
            hglayer.self_attn.v_proj.weight.copy_(v_weight.reshape(-1, hidden_size))

            hglayer.self_attn.o_proj.weight.copy_(mglayer.self_attention.linear_proj.weight)
            if num_experts is None:
                gate_weight, fc1_weight = torch.split(mglayer.mlp.linear_fc1.weight,
                                                      split_size_or_sections=args.ffn_hidden_size)
                hglayer.mlp.gate_proj.weight.copy_(gate_weight)
                hglayer.mlp.up_proj.weight.copy_(fc1_weight)
                hglayer.mlp.down_proj.weight.copy_(mglayer.mlp.linear_fc2.weight)
                hglayer.post_attention_layernorm.weight.copy_(mglayer.mlp.linear_fc1.layer_norm_weight)
            else:
                hglayer.post_attention_layernorm.weight.copy_(mglayer.pre_mlp_layernorm.weight)
                hglayer.mlp.gate.weight.copy_(mglayer.mlp.router.weight)
                if args.num_expert_split_size == 1:
                    for mgexpert, hgexpert in zip(mglayer.mlp.experts.local_experts, hglayer.mlp.experts):
                        gate_weight, fc1_weight = torch.split(mgexpert.linear_fc1.weight,
                                                              split_size_or_sections=args.ffn_hidden_size)
                        hgexpert.w1.weight.copy_(gate_weight)
                        hgexpert.w3.weight.copy_(fc1_weight)
                        hgexpert.w2.weight.copy_(mgexpert.linear_fc2.weight)
                else:
                    for mgexpert, hgexpert in zip(mglayer.mlp.experts.local_experts, hglayer.mlp.experts):
                        gate_weight, fc1_weight = torch.split(mgexpert.linear_fc1.weight,
                                                              split_size_or_sections=args.ffn_hidden_size)
                        hgexpert.w1.weight.copy_(gate_weight)
                        hgexpert.w3.weight.copy_(fc1_weight)
                        hgexpert.w2.weight.copy_(mgexpert.linear_fc2.weight)

        hgmodel.model.norm.weight.copy_(mgmodel.decoder.final_layernorm.weight)
        hgmodel.lm_head.weight.copy_(mgmodel.output_layer.weight)


def convert_checkpoint_from_transformers_to_megatron(mgmodel, hgmodel, args, hf_config):
    num_query_groups = hf_config.num_key_value_heads
    hidden_dim = hf_config.hidden_size
    head_dim = hidden_dim // hf_config.num_attention_heads
    num_experts = args.num_experts
    with torch.no_grad():
        mgmodel.embedding.word_embeddings.weight.copy_(hgmodel.model.embed_tokens.weight)
        for mglayer, hglayer in zip(mgmodel.decoder.layers, hgmodel.model.layers):
            mglayer.self_attention.linear_qkv.layer_norm_weight.copy_(hglayer.input_layernorm.weight)

            q = hglayer.self_attn.q_proj.weight.view([num_query_groups, -1, head_dim, hidden_dim])
            k = hglayer.self_attn.k_proj.weight.view([num_query_groups, -1, head_dim, hidden_dim])
            v = hglayer.self_attn.v_proj.weight.view([num_query_groups, -1, head_dim, hidden_dim])
            qkv = torch.cat([q, k, v], dim=1).view(-1, hidden_dim).contiguous()
            mglayer.self_attention.linear_qkv.weight.copy_(qkv)

            mglayer.self_attention.linear_proj.weight.copy_(hglayer.self_attn.o_proj.weight)
            fc1_weight = torch.cat([hglayer.mlp.gate_proj.weight, hglayer.mlp.up_proj.weight])
            if num_experts is None:
                mglayer.mlp.linear_fc1.weight.copy_(fc1_weight)
                mglayer.mlp.linear_fc2.weight.copy_(hglayer.mlp.down_proj.weight)
                mglayer.mlp.linear_fc1.layer_norm_weight.copy_(hglayer.post_attention_layernorm.weight)
            else:
                if args.num_expert_split_size == 1:
                    mglayer.pre_mlp_layernorm.weight.copy_(hglayer.post_attention_layernorm.weight)
                    nn.init.normal_(mglayer.mlp.router.weight, mean=0, std=0.02)
                    for expert in mglayer.mlp.experts.local_experts:
                        expert.linear_fc1.weight.copy_(fc1_weight)
                        expert.linear_fc2.weight.copy_(hglayer.mlp.down_proj.weight)
                else:
                    mglayer.pre_mlp_layernorm.weight.copy_(hglayer.post_attention_layernorm.weight)
                    nn.init.normal_(mglayer.mlp.router.weight, mean=0, std=0.02)

                    split_size = hf_config.intermediate_size // args.num_expert_split_size
                    gate_proj_splits = torch.split(hglayer.mlp.gate_proj.weight, split_size_or_sections=split_size)
                    up_proj_splits = torch.split(hglayer.mlp.up_proj.weight, split_size_or_sections=split_size)
                    down_proj_splits = torch.split(hglayer.mlp.down_proj.weight, split_size_or_sections=split_size, dim=1)

                    for idx, expert in enumerate(mglayer.mlp.experts.local_experts):
                        expert.linear_fc1.weight.copy_(torch.cat([gate_proj_splits[idx], up_proj_splits[idx]]))
                        expert.linear_fc2.weight.copy_(down_proj_splits[idx])

                    """
                    for idx, expert in enumerate(mglayer.mlp.experts.local_experts):
                        base_linear_fc1 = torch.cat([gate_proj_splits[idx], up_proj_splits[idx]])
                        extra_linear_fc1 = torch.empty(32, base_linear_fc1.shape[1])
                        extra_linear_fc2 = torch.empty(base_linear_fc1.shape[1], 16)
                        nn.init.normal_(extra_linear_fc1, mean=0, std=0.02)
                        nn.init.normal_(extra_linear_fc2, mean=0, std=0.02)
                        expert.linear_fc1.weight.copy_(torch.cat([base_linear_fc1, extra_linear_fc1.to(torch.float16)]))
                        expert.linear_fc2.weight.copy_(torch.cat([down_proj_splits[idx], extra_linear_fc2.to(torch.float16)], dim=1))
                    """

        mgmodel.decoder.final_layernorm.weight.copy_(hgmodel.model.norm.weight)
        mgmodel.output_layer.weight.copy_(hgmodel.lm_head.weight)


def save_state_dict(args, model, checkpoint_name):
    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size
    state_dict = {}
    state_dict['args'] = args
    state_dict['checkpoint_version'] = 3.0
    state_dict['iteration'] = 0
    state_dict['model'] = model
    os.makedirs(os.path.dirname(checkpoint_name), exist_ok=True)
    print(f'save model part {checkpoint_name}')
    torch.save(clone_state_dict(state_dict), checkpoint_name)


def save_mgmodel(args, mgmodel, load_path, save_path):
    # Saving config and tokenzier files
    copy_huggingface_tokenizer(load_path, save_path)
    tracker_filepath = os.path.join(save_path, 'latest_checkpointed_iteration.txt')
    with open(tracker_filepath, "w") as f:
        f.write("release")

    head_dim = args.hidden_size // args.num_attention_heads
    group_per_split = args.num_query_groups // args.target_tensor_model_parallel_size
    full_model = mgmodel.state_dict_for_save_checkpoint()
    for k in list(full_model.keys()):
        if full_model[k] is None or "_extra_state" in k:
            full_model.pop(k)
    pattern = r'local_experts\.(\d+)\.'
    num_local_experts = args.num_experts // args.target_expert_model_parallel_size if args.num_experts else 0
    if (
            args.target_tensor_model_parallel_size == 1
            and args.target_pipeline_model_parallel_size == 1
            and args.target_expert_model_parallel_size == 1
    ):
        checkpoint_name = get_checkpoint_name(save_path, 0, True)
        save_state_dict(args, full_model, checkpoint_name)
    elif (
            args.target_tensor_model_parallel_size == 1
            and args.target_pipeline_model_parallel_size == 1
            and args.num_experts
            and args.num_experts % args.target_expert_model_parallel_size == 0
    ):
        for ep_rank in range(args.target_expert_model_parallel_size):
            model_split = {}
            checkpoint_name = get_checkpoint_name(save_path, 0, True, None, None, None, True, ep_rank)
            print(f'save ep_rank {ep_rank} model to {checkpoint_name}')
            for k, v in full_model.items():
                if 'local_experts' in k:
                    expert_rank = int(re.findall(pattern, k)[0])
                    if expert_rank // num_local_experts != ep_rank:
                        continue
                    expert_local_rank = expert_rank % args.target_expert_model_parallel_size
                    k = k.replace(f'local_experts.{expert_rank}', f'local_experts.{expert_local_rank}')
                model_split[k] = v
            save_state_dict(args, model_split, checkpoint_name)
    elif (
            args.target_tensor_model_parallel_size > 1
            and args.target_pipeline_model_parallel_size == 1
            and args.num_experts is None
    ):
        for tp_rank in range(args.target_tensor_model_parallel_size):
            model_split = {}
            checkpoint_name = get_checkpoint_name(save_path, 0, True, None, tp_rank)
            print(f'tensor_parallel, save model to {checkpoint_name}')
            for k, v in full_model.items():
                if not isinstance(v, torch.Tensor):
                    target_v = v
                elif 'linear_qkv.weight' in k and 'norm' not in k:
                    viewed = v.view(args.num_query_groups, -1, head_dim, args.hidden_size)
                    viewed = viewed[group_per_split * tp_rank: group_per_split * (tp_rank + 1)]
                    target_v = viewed.view(-1, args.hidden_size)
                elif 'linear_qkv.bias' in k and 'norm' not in k:
                    viewed = v.view(args.num_query_groups, -1, head_dim)
                    viewed = viewed[group_per_split * tp_rank: group_per_split * (tp_rank + 1)]
                    target_v = viewed.view(-1)
                elif 'linear_proj' in k or 'linear_fc2' in k:
                    seg = v.shape[1] // args.target_tensor_model_parallel_size
                    target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]
                elif 'embedding' in k or 'output_layer' in k:
                    seg = v.shape[0] // args.target_tensor_model_parallel_size
                    target_v = v[seg * tp_rank: seg * (tp_rank + 1)]
                elif 'linear_fc1' in k and 'norm' not in k:
                    viewed = v.view(-1, args.ffn_hidden_size, args.hidden_size)
                    seg = args.ffn_hidden_size // args.target_tensor_model_parallel_size
                    target_v = viewed[:, seg * tp_rank: seg * (tp_rank + 1), :].reshape(-1, args.hidden_size)
                else:
                    target_v = v
                model_split[k] = target_v
            save_state_dict(args, model_split, checkpoint_name)
    elif (
            args.target_tensor_model_parallel_size > 1
            and args.target_pipeline_model_parallel_size == 1
            and args.num_experts
            and args.num_experts % args.target_expert_model_parallel_size == 0
    ):
        for tp_rank in range(args.target_tensor_model_parallel_size):
            for ep_rank in range(args.target_expert_model_parallel_size):
                model_split = {}
                checkpoint_name = get_checkpoint_name(save_path, 0, True, None, tp_rank, None, True, ep_rank)
                for k, v in full_model.items():
                    if not isinstance(v, torch.Tensor):
                        target_v = v
                    elif 'linear_qkv.weight' in k and 'norm' not in k:
                        viewed = v.view(args.num_query_groups, -1, head_dim, args.hidden_size)
                        viewed = viewed[group_per_split * tp_rank: group_per_split * (tp_rank + 1)]
                        target_v = viewed.view(-1, args.hidden_size)
                    elif 'linear_qkv.bias' in k and 'norm' not in k:
                        viewed = v.view(args.num_query_groups, -1, head_dim)
                        viewed = viewed[group_per_split * tp_rank: group_per_split * (tp_rank + 1)]
                        target_v = viewed.view(-1)
                    elif 'linear_proj' in k:
                        seg = v.shape[1] // args.target_tensor_model_parallel_size
                        target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]
                    elif 'embedding' in k or 'output_layer' in k:
                        seg = v.shape[0] // args.target_tensor_model_parallel_size
                        target_v = v[seg * tp_rank: seg * (tp_rank + 1)]
                    elif 'local_experts' in k:
                        expert_rank = int(re.findall(pattern, k)[0])
                        if expert_rank // num_local_experts != ep_rank:
                            continue
                        expert_local_rank = expert_rank % num_local_experts
                        if 'linear_fc1' in k and 'norm' not in k:
                            viewed = v.view(-1, args.ffn_hidden_size, args.hidden_size)
                            seg = args.ffn_hidden_size // args.target_tensor_model_parallel_size
                            target_v = viewed[:, seg * tp_rank: seg * (tp_rank + 1), :].reshape(-1, args.hidden_size)
                        elif 'linear_fc2' in k:
                            seg = v.shape[1] // args.target_tensor_model_parallel_size
                            target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]
                        k = k.replace(f'local_experts.{expert_rank}', f'local_experts.{expert_local_rank}')
                    else:
                        target_v = v
                    model_split[k] = target_v
                save_state_dict(args, model_split, checkpoint_name)
    else:
        raise ValueError('not support pp convert')
    print(f'megatron model is save to {save_path}')


def save_hgmodel(args, model):
    output_state_dict = model.state_dict()
    max_shard_size = args.max_shard_size
    shards, index = shard_checkpoint(output_state_dict, max_shard_size=max_shard_size)
    os.makedirs(args.save_path, exist_ok=True)
    for shard_file, shard in shards.items():
        target_file = os.path.join(args.save_path, shard_file)
        print(f'huggingface model is save to {target_file}')
        torch.save(clone_state_dict(shard), target_file)

    if index is None:
        print(f"Model weights saved in {os.path.join(args.save_path, WEIGHTS_NAME)}")
    else:
        save_index_file = os.path.join(args.save_path, WEIGHTS_INDEX_NAME)
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
        print(
            f"The model is bigger than the maximum size per checkpoint ({args.max_shard_size}) and is going to be "
            f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
            f"index located at {save_index_file}."
        )


def check_mg_eg_forward(mgmodel, hgmodel, mgargs):
    hg_hiddens = [{} for _ in range(mgargs.num_layers)]
    mg_hiddens = [{} for _ in range(mgargs.num_layers)]

    head_dim = mgargs.hidden_size // mgargs.num_attention_heads
    hidden_size = mgargs.hidden_size

    def print_input_hook(module, args, kwargs, layer_idx, mode):
        frame, name = mode.split('-')
        if frame == 'hg':
            hg_hiddens[layer_idx][name] = args[0].transpose(0, 1)
        elif frame == 'mg' and 'layer' in mode:
            mg_hiddens[layer_idx][name] = kwargs.get('hidden_states')
        elif frame == 'mg':
            mg_hiddens[layer_idx][name] = args[0]

    def print_output_hook(module, args, kwargs, output, layer_idx, mode):
        frame, name = mode.split('-')
        if mode in ['hg-q_proj_out', 'hg-k_proj_out', 'hg-v_proj_out']:
            hg_hiddens[layer_idx][name] = output
            hg_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['hg-lmhead']:
            hg_hiddens[layer_idx][name] = output.transpose(0, 1)
            hg_hiddens[layer_idx][name + '_token'] = output.transpose(0, 1).max(dim=-1)[1]
            print(output.transpose(0, 1).max(dim=-1))
        elif mode == 'hg-attn_out':
            hg_hiddens[layer_idx][name] = output[0].transpose(0, 1)
        elif mode in ['mg-lmhead']:
            mg_hiddens[layer_idx][name] = output[0]
            mg_hiddens[layer_idx][name + '_token'] = output[0].max(dim=-1)[1]
            print(output[0].max(dim=-1))
        elif mode == 'mg-attn_out':
            mg_hiddens[layer_idx][name] = output[0]
        elif mode == 'mg-qkv':
            mixed_qkv = output[0]
            sq, b, _ = mixed_qkv.shape
            mixed_qkv = mixed_qkv.view(sq, b, mgargs.num_query_groups, -1)
            qh = mgargs.num_attention_heads // mgargs.num_query_groups
            qo, ko, vo = torch.split(mixed_qkv, [qh * head_dim, head_dim, head_dim], dim=3)
            qo = qo.reshape(b, -1, hidden_size)
            ko = ko.reshape(b, -1, hidden_size // qh)
            vo = vo.reshape(b, -1, hidden_size // qh)
            mg_hiddens[layer_idx]['q_proj_out'] = qo
            mg_hiddens[layer_idx]['k_proj_out'] = ko
            mg_hiddens[layer_idx]['v_proj_out'] = vo

            weight = module.weight.view(mgargs.num_query_groups, -1, head_dim, hidden_size)
            qw, kw, vw = weight.split([qh, 1, 1], dim=1)
            mg_hiddens[layer_idx]['q_proj_out_weight'] = qw.reshape(-1, hidden_size)
            mg_hiddens[layer_idx]['k_proj_out_weight'] = kw.reshape(-1, hidden_size // qh)
            mg_hiddens[layer_idx]['v_proj_out_weight'] = vw.reshape(-1, hidden_size // qh)

    hgmodel.lm_head.register_forward_hook(partial(print_output_hook, layer_idx=mgargs.num_layers - 1, mode='hg-lmhead'),
                                          with_kwargs=True)
    mgmodel.output_layer.register_forward_hook(
        partial(print_output_hook, layer_idx=mgargs.num_layers - 1, mode='mg-lmhead'), with_kwargs=True)

    for idx, layer in enumerate(hgmodel.model.layers):
        layer.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='hg-layer_in'), with_kwargs=True)
        layer.self_attn.o_proj.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='hg-o_proj_in'),
                                                         with_kwargs=True)
        layer.self_attn.q_proj.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='hg-q_proj_out'),
                                                     with_kwargs=True)
        layer.self_attn.k_proj.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='hg-k_proj_out'),
                                                     with_kwargs=True)
        layer.self_attn.v_proj.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='hg-v_proj_out'),
                                                     with_kwargs=True)
        layer.self_attn.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='hg-attn_out'),
                                              with_kwargs=True)

    for idx, layer in enumerate(mgmodel.decoder.layers):
        layer.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='mg-layer_in'), with_kwargs=True)
        layer.self_attention.linear_qkv.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='mg-qkv'),
                                                              with_kwargs=True)
        layer.self_attention.linear_proj.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode='mg-o_proj_in'), with_kwargs=True)
        layer.self_attention.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='mg-attn_out'),
                                                   with_kwargs=True)

    input_ids = torch.tensor([[1, 2, 3]]).long().cuda()
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(input_ids, -100, True, True, True)

    with torch.inference_mode():
        try:
            hgmodel.cuda()
            hgmodel(input_ids=input_ids)
        except torch.cuda.OutOfMemoryError:
            print('oom for huggingface model forward')
        hgmodel.cpu()
        del hgmodel

    with torch.inference_mode():
        try:
            mgmodel.cuda()
            mgmodel(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        except torch.cuda.OutOfMemoryError:
            print('oom for megatron model forward')
        mgmodel.cpu()
        del mgmodel

    epsilon = 1e-5
    for idx, (hgh, mgh) in enumerate(zip(hg_hiddens, mg_hiddens)):
        if len(hgh) != len(mgh):
            continue
        for k, hgv in hgh.items():
            mgv, hgv = mgh[k].cpu(), hgv.cpu()
            same_num = (hgv != mgv).sum()
            diff_num = ((hgv - mgv) > epsilon).sum()
            diff_max = (hgv - mgv).abs().max()
            print(f'layer:{idx}, {k}, diff: {same_num}, diff>{epsilon}:[{diff_num}/{hgv.numel()}] diff_max:{diff_max}')


def check_tokenizer_is_same(hgtokenizer, mgtokenizer):
    if transformers.__version__ <= '4.33.2':
        print('please update transformers')
        return

    if mgtokenizer is None:
        return

    conversation = [
        {"role": "user", "content": "what's your name"},
        {"role": "bot", "content": "cold"},
    ]
    hgres = hgtokenizer.apply_chat_template(conversation)
    mgres = mgtokenizer.apply_chat_template(conversation)
    for x, y in zip(hgres, mgres):
        assert x == y, 'tokenizer is different for huggingface and megatron'


def add_ckpt_args(parser):
    parser = get_patch_args(parser)
    parser = add_checkpointing_args(parser)
    parser = add_megatron_checkpoint_args(parser)
    parser = add_transformers_checkpoint_args(parser)
    return parser


def main():
    initialize_megatron(extra_args_provider=add_ckpt_args)
    args = get_args()
    hf_config, hf_model = create_huggingface_model(args)
    mg_model = create_megatron_model(args, hf_config)
    if args.convert_checkpoint_from_megatron_to_transformers:
        load_megatron_model(args, mg_model)
        convert_checkpoint_from_megatron_to_transformers(mg_model, hf_model, args)
        # check_mg_eg_forward(mg_model, hf_model, args)
        save_hgmodel(args, hf_model)
    else:
        hf_model.from_pretrained(args.load_path)
        convert_checkpoint_from_transformers_to_megatron(mg_model, hf_model, args, hf_config)
        # check_mg_eg_forward(mg_model, hf_model, args)
        save_mgmodel(args, mg_model, args.load_path, args.save_path)


if __name__ == "__main__":
    main()

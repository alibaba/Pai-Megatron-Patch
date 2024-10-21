import os
import re
import torch.nn as nn
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from megatron.training.initialize import initialize_megatron
from megatron.training import get_args
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, read_metadata

import sys
path_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.append(os.path.join(path_dir, "examples"))
from qwen1_5.pretrain_mcore_qwen import model_provider
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

def add_model_args(parser):

    parser.add_argument(
        "--target-tensor-model-parallel-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--target-expert-model-parallel-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--num-splits",
        type=int,
        default=1
    )

    parser.add_argument(
        "--hf-ckpt-path",
        type=str
    )

    return parser

def convert_checkpoint_from_transformers_to_megatron(hgmodel, mgmodel, args):
    num_attention_heads = args.num_attention_heads
    hidden_dim = args.hidden_size
    head_dim = hidden_dim // args.num_attention_heads

    with torch.no_grad():
        mgmodel.embedding.word_embeddings.weight.copy_(hgmodel.model.embed_tokens.weight)
        for mglayer, hglayer in zip(mgmodel.decoder.layers, hgmodel.model.layers):
            mglayer.self_attention.linear_qkv.layer_norm_weight.copy_(hglayer.input_layernorm.weight)

            q = hglayer.self_attn.q_proj.weight.view([num_attention_heads, -1, head_dim, hidden_dim])
            k = hglayer.self_attn.k_proj.weight.view([num_attention_heads, -1, head_dim, hidden_dim])
            v = hglayer.self_attn.v_proj.weight.view([num_attention_heads, -1, head_dim, hidden_dim])
            qkv = torch.cat([q, k, v], dim=1).view(-1, hidden_dim).contiguous()

            q_bias = hglayer.self_attn.q_proj.bias.view([num_attention_heads, -1])
            k_bias = hglayer.self_attn.k_proj.bias.view([num_attention_heads, -1])
            v_bias = hglayer.self_attn.v_proj.bias.view([num_attention_heads, -1])
            qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=1).view(-1).contiguous()
            mglayer.self_attention.linear_qkv.weight.copy_(qkv)
            mglayer.self_attention.linear_qkv.bias.copy_(qkv_bias)

            mglayer.self_attention.linear_proj.weight.copy_(hglayer.self_attn.o_proj.weight)

            mglayer.pre_mlp_layernorm.weight.copy_(hglayer.post_attention_layernorm.weight)

            nn.init.normal_(mglayer.mlp.router.weight, mean=0, std=0.02)

            split_size = args.ffn_hidden_size // args.num_splits
            gate_proj_splits = torch.split(hglayer.mlp.gate_proj.weight, split_size_or_sections=split_size)
            up_proj_splits = torch.split(hglayer.mlp.up_proj.weight, split_size_or_sections=split_size)
            down_proj_splits = torch.split(hglayer.mlp.down_proj.weight, split_size_or_sections=split_size, dim=1)
            extra_size = args.moe_ffn_hidden_size - split_size
            for idx, expert in enumerate(mglayer.mlp.experts.local_experts):
                base_linear_fc1 = torch.cat([gate_proj_splits[idx%args.num_splits], up_proj_splits[idx%args.num_splits]])
                extra_linear_fc1 = torch.empty(2*extra_size, base_linear_fc1.shape[1])
                extra_linear_fc2 = torch.empty(base_linear_fc1.shape[1], extra_size)
                nn.init.normal_(extra_linear_fc1, mean=0, std=0.02)
                nn.init.normal_(extra_linear_fc2, mean=0, std=0.02)
                expert.linear_fc1.weight.copy_(torch.cat([base_linear_fc1, extra_linear_fc1.to(torch.float16)]))
                expert.linear_fc2.weight.copy_(torch.cat([down_proj_splits[idx%args.num_splits],
                                                          extra_linear_fc2.to(torch.float16)], dim=1))

            nn.init.normal_(mglayer.mlp.shared_expert_gate.weight, mean=0, std=0.02)
            nn.init.normal_(mglayer.mlp.shared_expert.linear_fc1.weight, mean=0, std=0.02)
            nn.init.normal_(mglayer.mlp.shared_expert.linear_fc2.weight, mean=0, std=0.02)

        mgmodel.decoder.final_layernorm.weight.copy_(hgmodel.model.norm.weight)
        mgmodel.output_layer.weight.copy_(hgmodel.lm_head.weight)


def save_state_dict(args, model, checkpoint_name):
    state_dict = {}
    state_dict['args'] = args
    state_dict['checkpoint_version'] = 3.0
    state_dict['iteration'] = 0
    state_dict['model'] = model
    os.makedirs(os.path.dirname(checkpoint_name), exist_ok=True)
    print(f'save model part {checkpoint_name}')
    torch.save(clone_state_dict(state_dict), checkpoint_name)


def save_mgmodel(mgmodel, args):

    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.expert_model_parallel_size = args.target_expert_model_parallel_size


    os.makedirs(args.save, exist_ok=True)
    os.system("cp -rf " + args.load + "/config*.json " + args.save)
    os.system("cp -rf " + args.load + "/tokenizer* " + args.save)
    os.system("cp -rf " + args.load + "/vocab.json " + args.save)
    os.system("cp -rf " + args.load + "/merges.txt " + args.save)

    tracker_filepath = os.path.join(args.save, 'latest_checkpointed_iteration.txt')
    with open(tracker_filepath, "w") as f:
        f.write("release")

    head_dim = args.hidden_size // args.num_attention_heads
    group_per_split = args.num_attention_heads // args.tensor_model_parallel_size
    full_model = mgmodel.state_dict_for_save_checkpoint()
    for k in list(full_model.keys()):
        if full_model[k] is None or "_extra_state" in k:
            full_model.pop(k)
    pattern = r'local_experts\.(\d+)\.'
    num_local_experts = args.num_experts // args.expert_model_parallel_size if args.num_experts else 0
    if (
        args.tensor_model_parallel_size == 1
        and args.pipeline_model_parallel_size == 1
        and args.expert_model_parallel_size == 1
    ):
        checkpoint_name = get_checkpoint_name(args.save, 0, True)
        save_state_dict(args, full_model, checkpoint_name)
    elif (
        args.tensor_model_parallel_size == 1
        and args.pipeline_model_parallel_size == 1
        and args.expert_model_parallel_size >1
        and args.num_experts % args.expert_model_parallel_size == 0
    ):

        for ep_rank in range(args.expert_model_parallel_size):
            model_split = {}
            checkpoint_name = get_checkpoint_name(args.save, 0, True, None, None, None, True, ep_rank)
            print(f'save ep_rank {ep_rank} model to {checkpoint_name}')
            for k, v in full_model.items():
                if 'local_experts' in k:
                    expert_rank = int(re.findall(pattern, k)[0])
                    if expert_rank // num_local_experts != ep_rank:
                        continue
                    expert_local_rank = expert_rank % num_local_experts
                    k = k.replace(f'local_experts.{expert_rank}', f'local_experts.{expert_local_rank}')
                model_split[k] = v
            save_state_dict(args, model_split, checkpoint_name)
    elif (
        args.tensor_model_parallel_size > 1
        and args.pipeline_model_parallel_size == 1
        and args.num_experts % args.expert_model_parallel_size == 0
    ):
        for tp_rank in range(args.tensor_model_parallel_size):
            for ep_rank in range(args.expert_model_parallel_size):
                model_split = {}
                if args.expert_model_parallel_size >1:
                    checkpoint_name = get_checkpoint_name(args.save, 0, True, None, tp_rank, None, True, ep_rank)
                elif args.expert_model_parallel_size ==1:
                    checkpoint_name = get_checkpoint_name(args.save, 0, True, None, tp_rank, None, False)
                for k, v in full_model.items():
                    if not isinstance(v, torch.Tensor):
                        target_v = v
                    elif 'linear_qkv.weight' in k and 'norm' not in k:
                        viewed = v.view(args.num_attention_heads, -1, head_dim, args.hidden_size)
                        viewed = viewed[group_per_split*tp_rank : group_per_split*(tp_rank + 1)]
                        target_v = viewed.view(-1, args.hidden_size)
                    elif 'linear_qkv.bias' in k and 'norm' not in k:
                        viewed = v.view(args.num_attention_heads, -1, head_dim)
                        viewed = viewed[group_per_split*tp_rank : group_per_split*(tp_rank + 1)]
                        target_v = viewed.view(-1)
                    elif 'linear_proj' in k:
                        seg = v.shape[1] // args.tensor_model_parallel_size
                        target_v = v[:, seg*tp_rank : seg*(tp_rank + 1)]
                    elif 'embedding' in k or 'output_layer' in k:
                        seg = v.shape[0] // args.tensor_model_parallel_size
                        target_v = v[seg*tp_rank : seg*(tp_rank + 1)]
                    elif 'local_experts' in k:
                        expert_rank = int(re.findall(pattern, k)[0])
                        if expert_rank // num_local_experts != ep_rank:
                            continue
                        expert_local_rank = expert_rank % num_local_experts
                        if 'linear_fc1' in k and 'norm' not in k:
                            viewed = v.view(-1, args.moe_ffn_hidden_size, args.hidden_size)
                            seg = args.moe_ffn_hidden_size // args.tensor_model_parallel_size
                            target_v = viewed[:, seg*tp_rank: seg*(tp_rank+1), :].reshape(-1, args.hidden_size)
                        elif 'linear_fc2' in k:
                            seg = v.shape[1] // args.tensor_model_parallel_size
                            target_v = v[:, seg*tp_rank : seg*(tp_rank + 1)]
                        k = k.replace(f'local_experts.{expert_rank}', f'local_experts.{expert_local_rank}')
                    elif 'shared_expert' in k and 'gate' not in k:
                        if 'linear_fc1' in k:
                            viewed = v.view(-1, args.shared_moe_ffn_hidden_size, args.hidden_size)
                            seg = args.shared_moe_ffn_hidden_size // args.tensor_model_parallel_size
                            target_v = viewed[:, seg*tp_rank: seg*(tp_rank+1), :].reshape(-1, args.hidden_size)
                        elif 'linear_fc2' in k:
                            seg = v.shape[1] // args.tensor_model_parallel_size
                            target_v = v[:, seg*tp_rank : seg*(tp_rank + 1)]
                    else:
                        target_v = v
                    model_split[k] = target_v
                save_state_dict(args, model_split, checkpoint_name)
    else:
        raise ValueError('not support pp convert')

    print(f'megatron model is save to {args.save}')

def add_extra_args(parser):
    parser = get_patch_args(parser)
    parser = add_model_args(parser)
    return parser

def main():
    initialize_megatron(extra_args_provider=add_extra_args)
    args = get_args()
    config = AutoConfig.from_pretrained(args.load)
    hf_model = AutoModelForCausalLM.from_pretrained(args.load, trust_remote_code=True, torch_dtype=config.torch_dtype)
    mg_model = model_provider()
    convert_checkpoint_from_transformers_to_megatron(hf_model, mg_model, args)
    save_mgmodel(mg_model, args)

if __name__ == "__main__":
    main()

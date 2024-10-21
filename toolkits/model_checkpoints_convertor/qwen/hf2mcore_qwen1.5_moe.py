import os
import re
import json
import torch
from collections import defaultdict
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
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
        "--hf-ckpt-path",
        type=str
    )

    return parser


def name_to_expert_rank(key):
    pattern = r'local_experts\.(\d+)\.'
    expert_rank = int(re.findall(pattern, key)[0])
    return expert_rank

def load_megatron_model(args):

    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.expert_model_parallel_size = args.target_expert_model_parallel_size

    if args.tensor_model_parallel_size >1 and args.expert_model_parallel_size >1:
        args.sequence_parallel = True

    os.makedirs(args.save, exist_ok=True)
    os.system("cp -rf " + args.hf_ckpt_path + "/config*.json " + args.save)
    os.system("cp -rf " + args.hf_ckpt_path+ "/tokenizer* " + args.save)
    os.system("cp -rf " + args.hf_ckpt_path + "/vocab.json " + args.save)
    os.system("cp -rf " + args.hf_ckpt_path + "/merges.txt " + args.save)

    os.system("cp -rf " + args.hf_ckpt_path + "/config*.json " + args.load)
    os.system("cp -rf " + args.hf_ckpt_path+ "/tokenizer* " + args.load)
    os.system("cp -rf " + args.hf_ckpt_path + "/vocab.json " + args.load)
    os.system("cp -rf " + args.hf_ckpt_path + "/merges.txt " + args.load)

    model = model_provider()
    if args.fp16 or args.bf16:
        model = model.bfloat16()  # [Float16Module(model_module, args) for model_module in model]
    model_path = args.load
    tracker_filename = get_checkpoint_tracker_filename(model_path)
    iteration, release = read_metadata(tracker_filename)
    head_dim = args.hidden_size // args.num_attention_heads
    group_per_split = args.num_attention_heads // args.tensor_model_parallel_size
    num_local_experts = args.num_experts // args.expert_model_parallel_size
    state_dict = {}
    mid_state = defaultdict(list)
    if (
        args.tensor_model_parallel_size == 1
        and args.pipeline_model_parallel_size == 1
        and args.expert_model_parallel_size == 1
    ):
        checkpoint_name = get_checkpoint_name(model_path, iteration, release, None, None, None, None, None)
        state_dict = torch.load(checkpoint_name)['model']
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
    else:
        raise ValueError('not support yet')

    model.load_state_dict(state_dict, strict=False)
    return model


def convert_checkpoint_from_megatron_to_transformers(mgmodel, hgmodel, args):
    num_attention_heads = args.num_attention_heads
    hidden_dim = args.hidden_size
    head_dim = hidden_dim // args.num_attention_heads

    with torch.no_grad():
        hgmodel.model.embed_tokens.weight.copy_(mgmodel.embedding.word_embeddings.weight)
        for mglayer, hglayer in zip(mgmodel.decoder.layers, hgmodel.model.layers):
            hglayer.input_layernorm.weight.copy_(mglayer.self_attention.linear_qkv.layer_norm_weight)

            qkv_weight = mglayer.self_attention.linear_qkv.weight.view(num_attention_heads, -1, head_dim, hidden_dim)
            q_weight, k_weight, v_weight = torch.split(qkv_weight, split_size_or_sections=[1, 1, 1], dim=1)
            hglayer.self_attn.q_proj.weight.copy_(q_weight.reshape(-1, hidden_dim))
            hglayer.self_attn.k_proj.weight.copy_(k_weight.reshape(-1, hidden_dim))
            hglayer.self_attn.v_proj.weight.copy_(v_weight.reshape(-1, hidden_dim))

            qkv_bias = mglayer.self_attention.linear_qkv.bias.view(num_attention_heads, -1)

            q_bias, k_bias, v_bias = torch.split(qkv_bias, split_size_or_sections=[q_weight.shape[2],
                                                                                   k_weight.shape[2],
                                                                                   v_weight.shape[2]], dim=1)
            hglayer.self_attn.q_proj.bias.copy_(q_bias.reshape(-1))
            hglayer.self_attn.k_proj.bias.copy_(k_bias.reshape(-1))
            hglayer.self_attn.v_proj.bias.copy_(v_bias.reshape(-1))

            hglayer.self_attn.o_proj.weight.copy_(mglayer.self_attention.linear_proj.weight)
            hglayer.post_attention_layernorm.weight.copy_(mglayer.pre_mlp_layernorm.weight)

            hglayer.mlp.gate.weight.copy_(mglayer.mlp.router.weight)
            for mgexpert, hgexpert in zip(mglayer.mlp.experts.local_experts, hglayer.mlp.experts):
                gate_weight, up_weight = torch.split(mgexpert.linear_fc1.weight, split_size_or_sections=args.moe_ffn_hidden_size)
                hgexpert.gate_proj.weight.copy_(gate_weight)
                hgexpert.up_proj.weight.copy_(up_weight)
                hgexpert.down_proj.weight.copy_(mgexpert.linear_fc2.weight)

            hglayer.mlp.shared_expert_gate.weight.copy_(mglayer.mlp.shared_expert_gate.weight)
            shared_expert_gate_weight, shared_expert_up_weight = \
                torch.split(mglayer.mlp.shared_expert.linear_fc1.weight, split_size_or_sections=args.shared_moe_ffn_hidden_size)
            hglayer.mlp.shared_expert.gate_proj.weight.copy_(shared_expert_gate_weight)
            hglayer.mlp.shared_expert.up_proj.weight.copy_(shared_expert_up_weight)
            hglayer.mlp.shared_expert.down_proj.weight.copy_(mglayer.mlp.shared_expert.linear_fc2.weight)


        hgmodel.model.norm.weight.copy_(mgmodel.decoder.final_layernorm.weight)
        hgmodel.lm_head.weight.copy_(mgmodel.output_layer.weight)


def convert_checkpoint_from_transformers_to_megatron(hgmodel, mgmodel, args):

    if args.fp16:
        mgmodel = mgmodel.float16()
        hgmodel = hgmodel.float16()
    elif args.bf16:
        mgmodel = mgmodel.bfloat16()
        hgmodel = hgmodel.bfloat16()

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

            mglayer.mlp.router.weight.copy_(hglayer.mlp.gate.weight)
            for hf_expert, expert in zip(hglayer.mlp.experts, mglayer.mlp.experts.local_experts):
                fc1_weight = torch.cat([hf_expert.gate_proj.weight, hf_expert.up_proj.weight])
                expert.linear_fc1.weight.copy_(fc1_weight)
                expert.linear_fc2.weight.copy_(hf_expert.down_proj.weight)

            mglayer.mlp.shared_expert_gate.weight.copy_(hglayer.mlp.shared_expert_gate.weight)
            shared_fc1_weight = torch.cat([hglayer.mlp.shared_expert.gate_proj.weight, hglayer.mlp.shared_expert.up_proj.weight])
            mglayer.mlp.shared_expert.linear_fc1.weight.copy_(shared_fc1_weight)
            mglayer.mlp.shared_expert.linear_fc2.weight.copy_(hglayer.mlp.shared_expert.down_proj.weight)

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


def save_hgmodel(args, model):
    output_state_dict = model.state_dict()
    max_shard_size="10GB"
    shards, index = shard_checkpoint(output_state_dict, max_shard_size=max_shard_size)
    os.makedirs(args.save, exist_ok=True)
    for shard_file, shard in shards.items():
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

def add_extra_args(parser):
    parser = get_patch_args(parser)
    parser = add_model_args(parser)
    return parser


def check_mg_eg_forward(mgmodel, hgmodel, mgargs):
    hg_hiddens = [{} for _ in range(mgargs.num_layers)]
    mg_hiddens = [{} for _ in range(mgargs.num_layers)]

    vocab_size = mgargs.padded_vocab_size
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
        if mode in ['hg-lmhead']:
            hg_hiddens[layer_idx][name] = output.transpose(0, 1).reshape(-1, vocab_size)
            hg_hiddens[layer_idx][name + "_weight"] = module.weight
            hg_hiddens[layer_idx][name + '_token'] = output.transpose(0, 1).max(dim=-1)[1]
        elif mode in ['mg-lmhead']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, vocab_size)
            mg_hiddens[layer_idx][name + "_weight"] = module.weight
            mg_hiddens[layer_idx][name + '_token'] = output[0].max(dim=-1)[1]
        elif mode in ['hg-o_proj_out']:
            hg_hiddens[layer_idx][name] = output
            hg_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['mg-o_proj_out']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, hidden_size)
            mg_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['hg-attn_out']:
            hg_hiddens[layer_idx][name] = output[0].reshape(-1, hidden_size)
        elif mode in ['mg-attn_out']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, hidden_size)
        elif mode in ['hg-down_proj_out']:
            hg_hiddens[layer_idx][name] = output.reshape(-1, hidden_size)
            hg_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['mg-down_proj_out']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, hidden_size)
            mg_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['hg-shared_experts_down_proj_out']:
            hg_hiddens[layer_idx][name] = output.reshape(-1, hidden_size)
            hg_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['mg-shared_experts_down_proj_out']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, hidden_size)
            mg_hiddens[layer_idx][name + '_weight'] = module.weight

    hgmodel.lm_head.register_forward_hook(partial(print_output_hook, layer_idx=mgargs.num_layers - 1, mode='hg-lmhead'),
                                          with_kwargs=True)
    mgmodel.output_layer.register_forward_hook(
        partial(print_output_hook, layer_idx=mgargs.num_layers - 1, mode='mg-lmhead'), with_kwargs=True)

    for idx, layer in enumerate(hgmodel.model.layers):
        layer.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='hg-layer_in'), with_kwargs=True)

        layer.self_attn.o_proj.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='hg-o_proj_in'),
                                                         with_kwargs=True)

        layer.self_attn.o_proj.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='hg-o_proj_out'),
                                                     with_kwargs=True)

        layer.self_attn.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='hg-attn_out'),
                                              with_kwargs=True)

    for idx, layer in enumerate(mgmodel.decoder.layers):
        layer.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='mg-layer_in'), with_kwargs=True)

        layer.self_attention.linear_proj.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode='mg-o_proj_in'), with_kwargs=True)

        layer.self_attention.linear_proj.register_forward_hook(
            partial(print_output_hook, layer_idx=idx, mode='mg-o_proj_out'), with_kwargs=True)

        layer.self_attention.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='mg-attn_out'),
                                                   with_kwargs=True)

    input_ids = torch.tensor([[1, 2, 3]]).long().cuda()
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(input_ids, -100, True, True, True)
    print(hgmodel)
    print(mgmodel)
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
        assert len(hgh) == len(mgh)
        for k, hgv in hgh.items():
            mgv, hgv = mgh[k].cpu(), hgv.cpu()
            same_num = (hgv != mgv).sum()
            diff_num = ((hgv - mgv) > epsilon).sum()
            diff_max = (hgv - mgv).abs().max()
            print(f'layer:{idx}, {k}, diff: {same_num}, diff>{epsilon}:[{diff_num}/{hgv.numel()}] diff_max:{diff_max}')

def main():
    initialize_megatron(extra_args_provider=add_extra_args)
    args = get_args()

    if args.convert_checkpoint_from_megatron_to_transformers:
        mg_model = load_megatron_model(args)
        config = AutoConfig.from_pretrained(args.hf_ckpt_path)
        hf_model = AutoModelForCausalLM.from_config(config=config, torch_dtype=config.torch_dtype)
        convert_checkpoint_from_megatron_to_transformers(mg_model, hf_model, args)
        save_hgmodel(args, hf_model)
    else:
        config = AutoConfig.from_pretrained(args.load)
        hf_model = AutoModelForCausalLM.from_pretrained(args.load, trust_remote_code=True, torch_dtype=config.torch_dtype)
        mg_model = model_provider()
        convert_checkpoint_from_transformers_to_megatron(hf_model, mg_model, args)
        check_mg_eg_forward(mg_model, hf_model, args)
        save_mgmodel(mg_model, args)

if __name__ == "__main__":
    main()

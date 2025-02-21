import os
import re
import json
import torch
import torch.nn as nn
from collections import defaultdict
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
from mistral.pretrain_mcore_mistral import model_provider
from megatron_patch.arguments import get_patch_args
from toolkits.model_checkpoints_convertor.utils import (
    save_hfmodel,
    save_state_dict
)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

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
        required=True,
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



def create_huggingface_model(args):
    if args.num_experts is None:
        copy_huggingface_tokenizer(args.huggingface_model_path, args.save_path)
        config, model = build_huggingface_model(args.huggingface_model_path, args.params_dtype)
    else:
        copy_huggingface_tokenizer(args.huggingface_model_path, args.save_path, with_code=True)
        config, model = build_huggingface_model(args.save_path, args.params_dtype, random_init=True)

    return config, model.eval()


def create_megatron_model(args, hf_config):
    args.hidden_size = hf_config.hidden_size
    args.num_layers = hf_config.num_hidden_layers
    args.num_attention_heads = hf_config.num_attention_heads
    args.kv_channels = args.hidden_size // args.num_attention_heads
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
        code_path = os.path.join(cur_dir, 'hf_mistral_moe')
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
        state_dict = torch.load(checkpoint_name, weights_only=False)['model']
    elif (
            args.target_tensor_model_parallel_size == 1
            and args.target_pipeline_model_parallel_size == 1
            and args.num_experts
            and args.num_experts % args.target_expert_model_parallel_size == 0
    ):
        for ep_rank in range(args.target_expert_model_parallel_size):
            checkpoint_name = get_checkpoint_name(model_path, iteration, release, None, None, None, True, ep_rank)
            print(f'load {checkpoint_name}')
            split_state = torch.load(checkpoint_name, map_location="cpu", weights_only=False)['model']
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
            split_state = torch.load(checkpoint_name, map_location="cpu", weights_only=False)['model']
            for k, v in split_state.items():
                mid_state[k].append(v)
        for k, v in mid_state.items():
            if not isinstance(v[0], torch.Tensor) or 'norm' in k:
                target_v = v[0]
            elif 'extra_state' in k:
                target_v = None
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
                split_state = torch.load(checkpoint_name, map_location="cpu", weights_only=False)['model']
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
            elif 'extra_state' in k:
                target_v = None
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
                mglayer.pre_mlp_layernorm.weight.copy_(hglayer.post_attention_layernorm.weight)
                nn.init.normal_(mglayer.mlp.router.weight, mean=0, std=0.02)
                for expert in mglayer.mlp.experts.local_experts:
                    expert.linear_fc1.weight.copy_(fc1_weight)
                    expert.linear_fc2.weight.copy_(hglayer.mlp.down_proj.weight)
        mgmodel.decoder.final_layernorm.weight.copy_(hgmodel.model.norm.weight)
        mgmodel.output_layer.weight.copy_(hgmodel.lm_head.weight)

def save_mgmodel(args, mgmodel, load_path, save_path):

    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size

    if args.num_experts is not None:
        args.expert_model_parallel_size = args.target_expert_model_parallel_size

    # Saving config and tokenzier files
    copy_huggingface_tokenizer(load_path, save_path)
    tracker_filepath = os.path.join(save_path, 'latest_checkpointed_iteration.txt')
    with open(tracker_filepath, "w") as f:
        f.write("release")

    head_dim = args.hidden_size // args.num_attention_heads
    group_per_split = args.num_query_groups // args.target_tensor_model_parallel_size
    full_model = mgmodel.state_dict_for_save_checkpoint()
    for k in list(full_model.keys()):
        if 'extra_state' in k:
            # NOTE: since TE 1.14, fp8 metadata will be saved as tensor. 
            # Always drop these values in the MG ckpt to avoid potential issue.
            # This should work fine because fp8 metadata is not supported by HF ckpt.
            full_model[k] = None
        elif full_model[k] is None:
            full_model.pop(k)
    pattern = r'local_experts\.(\d+)\.'
    num_local_experts = args.num_experts // args.target_expert_model_parallel_size if args.num_experts else 0
    if (
            args.target_tensor_model_parallel_size == 1
            and args.target_pipeline_model_parallel_size == 1
            and args.target_expert_model_parallel_size == 1
    ):
        checkpoint_name = get_checkpoint_name(save_path, 0, True)
        save_state_dict(args, [full_model], checkpoint_name)
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
            save_state_dict(args, [model_split], checkpoint_name)
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
            save_state_dict(args, [model_split], checkpoint_name)
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
                save_state_dict(args, [model_split], checkpoint_name)
    else:
        raise ValueError('not support pp convert')
    print(f'megatron model is save to {save_path}')


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
        args.save = args.save_path
        args.save_safetensors = False
        save_hfmodel(args, hf_model)
    else:
        hf_model.from_pretrained(args.load_path)
        convert_checkpoint_from_transformers_to_megatron(mg_model, hf_model, args, hf_config)
        save_mgmodel(args, mg_model, args.load_path, args.save_path)


if __name__ == "__main__":
    main()

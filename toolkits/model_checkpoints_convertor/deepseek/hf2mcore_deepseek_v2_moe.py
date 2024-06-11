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
from deepseek_v2.pretrain_deepseek import model_provider
from megatron_patch.arguments import get_patch_args

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


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

    if args.tensor_model_parallel_size > 1 and args.expert_model_parallel_size > 1:
        args.sequence_parallel = True

    os.makedirs(args.save, exist_ok=True)
    os.system("cp -rf " + args.hf_ckpt_path + "/config*.json " + args.save)
    os.system("cp -rf " + args.hf_ckpt_path + "/tokenizer* " + args.save)
    os.system("cp -rf " + args.hf_ckpt_path + "/*.py " + args.save)

    os.system("cp -rf " + args.hf_ckpt_path + "/config*.json " + args.load)
    os.system("cp -rf " + args.hf_ckpt_path + "/tokenizer* " + args.load)
    os.system("cp -rf " + args.hf_ckpt_path + "/*.py " + args.load)

    model = model_provider()

    model_path = args.load
    tracker_filename = get_checkpoint_tracker_filename(model_path)
    iteration, release = read_metadata(tracker_filename)
    head_dim = args.hidden_size // args.num_attention_heads
    q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
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
            and args.expert_model_parallel_size > 1
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
                if args.expert_model_parallel_size > 1:
                    checkpoint_name = get_checkpoint_name(model_path, iteration, release, None, tp_rank, None, True,
                                                          ep_rank)
                elif args.expert_model_parallel_size == 1:
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
            if not isinstance(v[0], torch.Tensor) or 'norm' in k or 'router' in k or 'gate' in k or "linear_kv_a_proj_with_mqa" in k:
                target_v = v[0]
            elif 'embedding' in k or 'output_layer' in k:
                target_v = torch.cat(v, dim=0)
            elif 'linear_proj' in k or 'linear_fc2' in k:
                target_v = torch.cat(v, dim=1)
            elif 'linear_q_proj' in k:
                viewed = [x.view(group_per_split, -1, q_head_dim, args.hidden_size) for x in v]
                target_v = torch.cat(viewed, dim=0).view(-1, args.hidden_size)
            elif 'linear_kv_b_proj' in k:
                viewed = [x.view(group_per_split, -1, q_head_dim - args.qk_rope_head_dim + args.v_head_dim, args.kv_lora_rank) for x in v]
                target_v = torch.cat(viewed, dim=0).view(-1, args.kv_lora_rank)
            elif 'linear_fc1' in k:
                viewed = [x.view(2, -1, args.hidden_size) for x in v]
                target_v = torch.cat(viewed, dim=1).view(-1, args.hidden_size)
            else:
                print('passed', k)
                exit()
            state_dict[k] = target_v
    else:
        raise ValueError('not support yet')

    model.load_state_dict(state_dict)
    return model


def convert_checkpoint_from_megatron_to_transformers(mgmodel, hgmodel, args):
    num_attention_heads = args.num_attention_heads
    hidden_dim = args.hidden_size
    head_dim = hidden_dim // args.num_attention_heads

    with torch.no_grad():
        hgmodel.model.embed_tokens.weight.copy_(mgmodel.embedding.word_embeddings.weight)
        for layer_idx, (mglayer, hglayer) in enumerate(zip(mgmodel.decoder.layers, hgmodel.model.layers)):
            # mglayer.input_layernorm.weight.copy_(hglayer.input_layernorm.weight)
            hglayer.input_layernorm.weight.copy_(mglayer.input_layernorm.weight)

            # mglayer.pre_mlp_layernorm.weight.copy_(hglayer.post_attention_layernorm.weight)
            hglayer.post_attention_layernorm.weight.copy_(mglayer.pre_mlp_layernorm.weight)

            # mglayer.self_attention.linear_q_proj.weight.copy_(hglayer.self_attn.q_proj.weight)
            hglayer.self_attn.q_proj.weight.copy_(mglayer.self_attention.linear_q_proj.weight)

            # mglayer.self_attention.linear_kv_a_proj_with_mqa.weight.copy_(hglayer.self_attn.kv_a_proj_with_mqa.weight)
            hglayer.self_attn.kv_a_proj_with_mqa.weight.copy_(mglayer.self_attention.linear_kv_a_proj_with_mqa.weight)

            # mglayer.self_attention.linear_kv_b_proj.weight.copy_(hglayer.self_attn.kv_b_proj.weight)
            hglayer.self_attn.kv_b_proj.weight.copy_(mglayer.self_attention.linear_kv_b_proj.weight)

            # mglayer.self_attention.kv_a_layernorm.weight.copy_(hglayer.self_attn.kv_a_layernorm.weight)
            hglayer.self_attn.kv_a_layernorm.weight.copy_(mglayer.self_attention.kv_a_layernorm.weight)

            # mglayer.self_attention.linear_proj.weight.copy_(hglayer.self_attn.o_proj.weight)
            hglayer.self_attn.o_proj.weight.copy_(mglayer.self_attention.linear_proj.weight)

            if layer_idx == 0:

                #mglayer.mlp.linear_fc1.weight.copy_(torch.cat([hglayer.mlp.gate_proj.weight, hglayer.mlp.up_proj.weight]))
                gate_weight, up_weight = torch.split(mglayer.mlp.linear_fc1.weight, split_size_or_sections=args.ffn_hidden_size)
                hglayer.mlp.gate_proj.weight.copy_(gate_weight)
                hglayer.mlp.up_proj.weight.copy_(up_weight)

                #mglayer.mlp.linear_fc2.weight.copy_(hglayer.mlp.down_proj.weight)
                hglayer.mlp.down_proj.weight.copy_(mglayer.mlp.linear_fc2.weight)

            else:
                # mglayer.mlp.router.weight.copy_(hglayer.mlp.gate.weight)
                hglayer.mlp.gate.weight.copy_(mglayer.mlp.router.weight)

                for mgexpert, hgexpert in zip(mglayer.mlp.experts.local_experts, hglayer.mlp.experts):
                    gate_weight, up_weight = torch.split(mgexpert.linear_fc1.weight,
                                                         split_size_or_sections=args.moe_ffn_hidden_size)
                    hgexpert.gate_proj.weight.copy_(gate_weight)
                    hgexpert.up_proj.weight.copy_(up_weight)
                    hgexpert.down_proj.weight.copy_(mgexpert.linear_fc2.weight)

                shared_expert_gate_weight, shared_expert_up_weight = \
                    torch.split(mglayer.mlp.shared_expert.linear_fc1.weight,
                                split_size_or_sections=args.moe_ffn_hidden_size*args.num_shared_experts)
                hglayer.mlp.shared_experts.gate_proj.weight.copy_(shared_expert_gate_weight)
                hglayer.mlp.shared_experts.up_proj.weight.copy_(shared_expert_up_weight)
                hglayer.mlp.shared_experts.down_proj.weight.copy_(mglayer.mlp.shared_expert.linear_fc2.weight)

        hgmodel.model.norm.weight.copy_(mgmodel.decoder.final_layernorm.weight)
        hgmodel.lm_head.weight.copy_(mgmodel.output_layer.weight)


def convert_checkpoint_from_transformers_to_megatron(hgmodel, mgmodel, args):
    if args.fp16:
        mgmodel = mgmodel.float16()
        hgmodel = hgmodel.float16()
    elif args.bf16:
        mgmodel = mgmodel.bfloat16()
        hgmodel = hgmodel.bfloat16()

    with torch.no_grad():
        mgmodel.embedding.word_embeddings.weight.copy_(hgmodel.model.embed_tokens.weight)
        for layer_idx, (mglayer, hglayer) in enumerate(zip(mgmodel.decoder.layers, hgmodel.model.layers)):
            mglayer.input_layernorm.weight.copy_(hglayer.input_layernorm.weight)
            mglayer.pre_mlp_layernorm.weight.copy_(hglayer.post_attention_layernorm.weight)
            if args.q_lora_rank is not None:
                mglayer.self_attention.linear_q_a_proj.weight.copy_(hglayer.self_attn.q_a_proj.weight)
                mglayer.self_attention.linear_q_b_proj.weight.copy_(hglayer.self_attn.q_b_proj.weight)
                mglayer.self_attention.q_a_layernorm.weight.copy_(hglayer.self_attn.q_a_layernorm.weight)
            else:
                mglayer.self_attention.linear_q_proj.weight.copy_(hglayer.self_attn.q_proj.weight)
            mglayer.self_attention.linear_kv_a_proj_with_mqa.weight.copy_(hglayer.self_attn.kv_a_proj_with_mqa.weight)
            mglayer.self_attention.linear_kv_b_proj.weight.copy_(hglayer.self_attn.kv_b_proj.weight)
            mglayer.self_attention.kv_a_layernorm.weight.copy_(hglayer.self_attn.kv_a_layernorm.weight)
            mglayer.self_attention.linear_proj.weight.copy_(hglayer.self_attn.o_proj.weight)

            if layer_idx == 0:
                mglayer.mlp.linear_fc1.weight.copy_(
                    torch.cat([hglayer.mlp.gate_proj.weight, hglayer.mlp.up_proj.weight]))
                mglayer.mlp.linear_fc2.weight.copy_(hglayer.mlp.down_proj.weight)
            else:
                mglayer.mlp.router.weight.copy_(hglayer.mlp.gate.weight)
                for hf_expert, expert in zip(hglayer.mlp.experts, mglayer.mlp.experts.local_experts):
                    fc1_weight = torch.cat([hf_expert.gate_proj.weight, hf_expert.up_proj.weight])
                    expert.linear_fc1.weight.copy_(fc1_weight)
                    expert.linear_fc2.weight.copy_(hf_expert.down_proj.weight)

                shared_fc1_weight = torch.cat(
                    [hglayer.mlp.shared_experts.gate_proj.weight, hglayer.mlp.shared_experts.up_proj.weight])
                mglayer.mlp.shared_expert.linear_fc1.weight.copy_(shared_fc1_weight)
                mglayer.mlp.shared_expert.linear_fc2.weight.copy_(hglayer.mlp.shared_experts.down_proj.weight)

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
    torch.save(state_dict, checkpoint_name)


def save_mgmodel(mgmodel, args):
    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.expert_model_parallel_size = args.target_expert_model_parallel_size

    os.makedirs(args.save, exist_ok=True)
    os.system("cp -rf " + args.load + "/config*.json " + args.save)
    os.system("cp -rf " + args.load + "/tokenizer* " + args.save)

    tracker_filepath = os.path.join(args.save, 'latest_checkpointed_iteration.txt')
    with open(tracker_filepath, "w") as f:
        f.write("release")

    q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
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
            and args.expert_model_parallel_size > 1
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
                if args.expert_model_parallel_size > 1:
                    checkpoint_name = get_checkpoint_name(args.save, 0, True, None, tp_rank, None, True, ep_rank)
                elif args.expert_model_parallel_size == 1:
                    checkpoint_name = get_checkpoint_name(args.save, 0, True, None, tp_rank, None, False)
                for k, v in full_model.items():
                    if not isinstance(v, torch.Tensor):
                        target_v = v
                    elif 'linear_q_proj' in k or 'linear_q_a_proj' in k:
                        seg = v.shape[0] // args.tensor_model_parallel_size
                        target_v = v[seg * tp_rank: seg * (tp_rank + 1)]
                    elif 'linear_q_b_proj' in k:
                        seg_0 = v.shape[0] // args.tensor_model_parallel_size
                        seg_1 = v.shape[1] // args.tensor_model_parallel_size
                        target_v = v[seg_0 * tp_rank: seg_0 * (tp_rank + 1), seg_1 * tp_rank: seg_1 * (tp_rank + 1)]
                    elif 'q_a_layernorm' in k:
                        seg = v.shape[0] // args.tensor_model_parallel_size
                        target_v = v[seg * tp_rank: seg * (tp_rank + 1)]
                    elif 'linear_kv_b_proj' in k:
                        seg = v.shape[0] // args.tensor_model_parallel_size
                        target_v = v[seg * tp_rank:seg* (tp_rank + 1)]
                    elif 'linear_proj' in k:
                        seg = v.shape[1] // args.tensor_model_parallel_size
                        target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]
                    elif 'embedding' in k or 'output_layer' in k:
                        seg = v.shape[0] // args.tensor_model_parallel_size
                        target_v = v[seg * tp_rank: seg * (tp_rank + 1)]
                    elif 'decoder.layers.0.mlp.linear_fc2' in k:
                        seg = v.shape[1] // args.tensor_model_parallel_size
                        target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]
                    elif 'decoder.layers.0.mlp.linear_fc1' in k:
                        viewed = v.view(-1, args.ffn_hidden_size, args.hidden_size)
                        seg = args.ffn_hidden_size // args.tensor_model_parallel_size
                        target_v = viewed[:, seg * tp_rank: seg * (tp_rank + 1), :].reshape(-1, args.hidden_size)
                    elif 'local_experts' in k:
                        expert_rank = int(re.findall(pattern, k)[0])
                        if expert_rank // num_local_experts != ep_rank:
                            continue
                        expert_local_rank = expert_rank % num_local_experts
                        if 'linear_fc1' in k and 'norm' not in k:
                            viewed = v.view(-1, args.moe_ffn_hidden_size, args.hidden_size)
                            seg = args.moe_ffn_hidden_size // args.tensor_model_parallel_size
                            target_v = viewed[:, seg * tp_rank: seg * (tp_rank + 1), :].reshape(-1, args.hidden_size)
                        elif 'linear_fc2' in k:
                            seg = v.shape[1] // args.tensor_model_parallel_size
                            target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]
                        k = k.replace(f'local_experts.{expert_rank}', f'local_experts.{expert_local_rank}')
                    elif 'shared_expert' in k and 'gate' not in k:
                        if 'linear_fc1' in k:
                            viewed = v.view(-1, args.moe_ffn_hidden_size * args.num_shared_experts, args.hidden_size)
                            seg = args.moe_ffn_hidden_size * args.num_shared_experts // args.tensor_model_parallel_size
                            target_v = viewed[:, seg * tp_rank: seg * (tp_rank + 1), :].reshape(-1, args.hidden_size)
                        elif 'linear_fc2' in k:
                            seg = v.shape[1] // args.tensor_model_parallel_size
                            target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]
                    else:
                        target_v = v
                    model_split[k] = target_v
                save_state_dict(args, model_split, checkpoint_name)
    else:
        raise ValueError('not support pp convert')

    print(f'megatron model is save to {args.save}')


def save_hgmodel(args, model):
    output_state_dict = model.state_dict()
    max_shard_size = "10GB"
    shards, index = shard_checkpoint(output_state_dict, max_shard_size=max_shard_size)
    os.makedirs(args.save, exist_ok=True)
    for shard_file, shard in shards.items():
        target_file = os.path.join(args.save, shard_file)
        print(f'huggingface model is save to {target_file}')
        torch.save(shard, target_file)

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

    hidden_size = mgargs.hidden_size
    q_head_dim = mgargs.qk_nope_head_dim + mgargs.qk_rope_head_dim
    num_heads = mgargs.num_attention_heads
    v_head_dim = mgargs.v_head_dim
    vocab_size = mgargs.padded_vocab_size
    kv_a_dim = mgargs.kv_lora_rank + mgargs.qk_rope_head_dim
    kv_b_dim = num_heads * (q_head_dim - mgargs.qk_rope_head_dim + v_head_dim)
    kv_lora_rank = mgargs.kv_lora_rank

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
        elif mode in ['hg-q_proj_out', 'hg-o_proj_out', 'hg-kv_b_proj_out', 'hg-kv_a_proj_out', 'hg-kv_a_norm_out']:
            hg_hiddens[layer_idx][name] = output
            hg_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['mg-kv_a_norm_out']:
            mg_hiddens[layer_idx][name] = output.reshape(-1, kv_lora_rank)
            mg_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['mg-q_proj_out']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, num_heads * q_head_dim)
            mg_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['mg-kv_a_proj_out']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, kv_a_dim)
            mg_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['mg-kv_b_proj_out']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, kv_b_dim)
            mg_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['mg-o_proj_out']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, num_heads * v_head_dim)
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

        if mgargs.q_lora_rank is None:

            layer.self_attn.q_proj.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='hg-q_proj_in'),
                                                             with_kwargs=True)

            layer.self_attn.q_proj.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='hg-q_proj_out'),
                                                         with_kwargs=True)

        layer.self_attn.kv_a_proj_with_mqa.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode='hg-kv_a_proj_in'), with_kwargs=True)

        layer.self_attn.kv_a_proj_with_mqa.register_forward_hook(
            partial(print_output_hook, layer_idx=idx, mode='hg-kv_a_proj_out'), with_kwargs=True)

        layer.self_attn.kv_a_layernorm.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode='hg-kv_a_norm_in'), with_kwargs=True)

        layer.self_attn.kv_a_layernorm.register_forward_hook(
            partial(print_output_hook, layer_idx=idx, mode='hg-kv_a_norm_out'), with_kwargs=True)

        layer.self_attn.kv_b_proj.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode='hg-kv_b_proj_in'), with_kwargs=True)

        layer.self_attn.kv_b_proj.register_forward_hook(
            partial(print_output_hook, layer_idx=idx, mode='hg-kv_b_proj_out'), with_kwargs=True)

        layer.self_attn.o_proj.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='hg-o_proj_in'),
                                                         with_kwargs=True)

        layer.self_attn.o_proj.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='hg-o_proj_out'),
                                                     with_kwargs=True)

        layer.self_attn.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='hg-attn_out'),
                                              with_kwargs=True)

        if idx == 0:
            layer.mlp.down_proj.register_forward_pre_hook(
                partial(print_input_hook, layer_idx=idx, mode='hg-down_proj_in'), with_kwargs=True)

            layer.mlp.down_proj.register_forward_hook(
                partial(print_output_hook, layer_idx=idx, mode='hg-down_proj_out'), with_kwargs=True)
        else:
            layer.mlp.shared_experts.down_proj.register_forward_pre_hook(
                partial(print_input_hook, layer_idx=idx, mode='hg-shared_experts_down_proj_in'), with_kwargs=True)

            layer.mlp.shared_experts.down_proj.register_forward_hook(
                partial(print_output_hook, layer_idx=idx, mode='hg-shared_experts_down_proj_out'), with_kwargs=True)

    for idx, layer in enumerate(mgmodel.decoder.layers):

        layer.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='mg-layer_in'), with_kwargs=True)

        if mgargs.q_lora_rank is None:
            layer.self_attention.linear_q_proj.register_forward_pre_hook(
                partial(print_input_hook, layer_idx=idx, mode='mg-q_proj_in'), with_kwargs=True)

            layer.self_attention.linear_q_proj.register_forward_hook(
                partial(print_output_hook, layer_idx=idx, mode='mg-q_proj_out'), with_kwargs=True)

        layer.self_attention.linear_kv_a_proj_with_mqa.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode='mg-kv_a_proj_in'), with_kwargs=True)

        layer.self_attention.linear_kv_a_proj_with_mqa.register_forward_hook(
            partial(print_output_hook, layer_idx=idx, mode='mg-kv_a_proj_out'), with_kwargs=True)

        layer.self_attention.kv_a_layernorm.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode='mg-kv_a_norm_in'), with_kwargs=True)

        layer.self_attention.kv_a_layernorm.register_forward_hook(
            partial(print_output_hook, layer_idx=idx, mode='mg-kv_a_norm_out'), with_kwargs=True)

        layer.self_attention.linear_kv_b_proj.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode='mg-kv_b_proj_in'), with_kwargs=True)

        layer.self_attention.linear_kv_b_proj.register_forward_hook(
            partial(print_output_hook, layer_idx=idx, mode='mg-kv_b_proj_out'), with_kwargs=True)

        layer.self_attention.linear_proj.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode='mg-o_proj_in'), with_kwargs=True)

        layer.self_attention.linear_proj.register_forward_hook(
            partial(print_output_hook, layer_idx=idx, mode='mg-o_proj_out'), with_kwargs=True)

        layer.self_attention.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='mg-attn_out'),
                                                   with_kwargs=True)

        if idx == 0:
            layer.mlp.linear_fc2.register_forward_pre_hook(
                partial(print_input_hook, layer_idx=idx, mode='mg-down_proj_in'), with_kwargs=True)

            layer.mlp.linear_fc2.register_forward_hook(
                partial(print_output_hook, layer_idx=idx, mode='mg-down_proj_out'), with_kwargs=True)
        else:
            layer.mlp.shared_expert.linear_fc2.register_forward_pre_hook(
                partial(print_input_hook, layer_idx=idx, mode='mg-shared_experts_down_proj_in'), with_kwargs=True)

            layer.mlp.shared_expert.linear_fc2.register_forward_hook(
                partial(print_output_hook, layer_idx=idx, mode='mg-shared_experts_down_proj_out'), with_kwargs=True)

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
        hf_model = AutoModelForCausalLM.from_pretrained(args.hf_ckpt_path, trust_remote_code=True)
        mg_model = load_megatron_model(args)
        convert_checkpoint_from_megatron_to_transformers(mg_model, hf_model, args)
        save_hgmodel(args, hf_model)
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(args.load, trust_remote_code=True)
        mg_model = model_provider()
        convert_checkpoint_from_transformers_to_megatron(hf_model, mg_model, args)
        if args.q_lora_rank is None:
            check_mg_eg_forward(mg_model, hf_model, args)
        save_mgmodel(mg_model, args)


if __name__ == "__main__":
    main()

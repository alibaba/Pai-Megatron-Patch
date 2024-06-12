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
from qwen2.pretrain_qwen import model_provider
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
        "--hf-ckpt-path",
        type=str
    )

    return parser


def load_megatron_model(args):
    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size

    if args.tensor_model_parallel_size >1:
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


    model_path = args.load
    tracker_filename = get_checkpoint_tracker_filename(model_path)
    iteration, release = read_metadata(tracker_filename)
    head_dim = args.hidden_size // args.num_attention_heads
    group_per_split = args.num_query_groups // args.tensor_model_parallel_size

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
    else:
        raise ValueError('not support yet')

    model.load_state_dict(state_dict)
    return model


def convert_checkpoint_from_megatron_to_transformers(mgmodel, hfmodel, args):

    if args.fp16:
        mgmodel = mgmodel.float16()
    elif args.bf16:
        mgmodel = mgmodel.bfloat16()

    num_query_groups = args.num_query_groups
    hidden_size = args.hidden_size
    head_dim = hidden_size // args.num_attention_heads
    use_te = args.transformer_impl == "transformer_engine"
    value_num_per_group = args.num_attention_heads // num_query_groups
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

            q_bias, k_bias, v_bias = torch.split(mglayer.self_attention.linear_qkv.bias,
                                                 split_size_or_sections=[hflayer.self_attn.q_proj.weight.shape[0],
                                                                         hflayer.self_attn.k_proj.weight.shape[0],
                                                                         hflayer.self_attn.v_proj.weight.shape[0]], dim=0)
            hflayer.self_attn.q_proj.bias.copy_(q_bias.reshape(-1))
            hflayer.self_attn.k_proj.bias.copy_(k_bias.reshape(-1))
            hflayer.self_attn.v_proj.bias.copy_(v_bias.reshape(-1))

            hflayer.self_attn.o_proj.weight.copy_(mglayer.self_attention.linear_proj.weight)

            gate_weight, fc1_weight = torch.split(mglayer.mlp.linear_fc1.weight, split_size_or_sections=args.ffn_hidden_size)
            hflayer.mlp.gate_proj.weight.copy_(gate_weight)
            hflayer.mlp.up_proj.weight.copy_(fc1_weight)
            hflayer.mlp.down_proj.weight.copy_(mglayer.mlp.linear_fc2.weight)
            if use_te:
                hflayer.post_attention_layernorm.weight.copy_(mglayer.mlp.linear_fc1.layer_norm_weight)
            else:
                hflayer.post_attention_layernorm.weight.copy_(mglayer.pre_mlp_layernorm.weight)

        hfmodel.model.norm.weight.copy_(mgmodel.decoder.final_layernorm.weight)
        hfmodel.lm_head.weight.copy_(mgmodel.output_layer.weight)


def convert_checkpoint_from_transformers_to_megatron(hfmodel, mgmodel, args):

    if args.fp16:
        mgmodel = mgmodel.float16()
    elif args.bf16:
        mgmodel = mgmodel.bfloat16()

    num_query_groups = args.num_query_groups
    hidden_size = args.hidden_size
    head_dim = hidden_size // args.num_attention_heads
    use_te = args.transformer_impl == "transformer_engine"

    with torch.no_grad():
        mgmodel.embedding.word_embeddings.weight.copy_(hfmodel.model.embed_tokens.weight)
        for mglayer, hflayer in zip(mgmodel.decoder.layers, hfmodel.model.layers):
            if use_te:
                mglayer.self_attention.linear_qkv.layer_norm_weight.copy_(hflayer.input_layernorm.weight)
            else:
                mglayer.input_layernorm.weight.copy_(hflayer.input_layernorm.weight)

            q = hflayer.self_attn.q_proj.weight.view([num_query_groups, -1, head_dim, hidden_size])
            k = hflayer.self_attn.k_proj.weight.view([num_query_groups, -1, head_dim, hidden_size])
            v = hflayer.self_attn.v_proj.weight.view([num_query_groups, -1, head_dim, hidden_size])

            qkv = torch.cat([q, k, v], dim=1).view(-1, hidden_size).contiguous()
            q_bias = hflayer.self_attn.q_proj.bias.view([num_query_groups, -1])
            k_bias = hflayer.self_attn.k_proj.bias.view([num_query_groups, -1])
            v_bias = hflayer.self_attn.v_proj.bias.view([num_query_groups, -1])
            qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=1).view(-1).contiguous()
            mglayer.self_attention.linear_qkv.weight.copy_(qkv)
            mglayer.self_attention.linear_qkv.bias.copy_(qkv_bias)

            mglayer.self_attention.linear_proj.weight.copy_(hflayer.self_attn.o_proj.weight)
            fc1_weight = torch.cat([hflayer.mlp.gate_proj.weight, hflayer.mlp.up_proj.weight])
            mglayer.mlp.linear_fc1.weight.copy_(fc1_weight)
            mglayer.mlp.linear_fc2.weight.copy_(hflayer.mlp.down_proj.weight)
            if use_te:
                mglayer.mlp.linear_fc1.layer_norm_weight.copy_(hflayer.post_attention_layernorm.weight)
            else:
                mglayer.pre_mlp_layernorm.weight.copy_(hflayer.post_attention_layernorm.weight)

        mgmodel.decoder.final_layernorm.weight.copy_(hfmodel.model.norm.weight)
        mgmodel.output_layer.weight.copy_(hfmodel.lm_head.weight)


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
        for tp_rank in range(args.tensor_model_parallel_size):
            model_split = {}
            checkpoint_name = get_checkpoint_name(args.save, 0, True, None, tp_rank)
            print(f'tensor_parallel, save model to {checkpoint_name}')
            for k, v in full_model.items():
                if not isinstance(v, torch.Tensor):
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
                model_split[k] = target_v
            save_state_dict(args, model_split, checkpoint_name)
    else:
        raise ValueError('not support pp convert')
    print(f'megatron model is save to {args.save}')


def save_hfmodel(args, model):
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

def check_hf_mg_forward(hfmodel, mgmodel, mgargs):
    hf_hiddens = [{} for _ in range(mgargs.num_layers)]
    mg_hiddens = [{} for _ in range(mgargs.num_layers)]

    hidden_size = mgargs.hidden_size
    vocab_size = mgargs.padded_vocab_size


    def print_input_hook(module, args, kwargs, layer_idx, mode):
        frame, name = mode.split('-')
        if frame == 'hf':
            hf_hiddens[layer_idx][name] = args[0].transpose(0, 1)
        elif frame == 'mg' and 'layer' in mode:
            mg_hiddens[layer_idx][name] = kwargs.get('hidden_states')
        elif frame == 'mg':
            mg_hiddens[layer_idx][name] = args[0]

    def print_output_hook(module, args, kwargs, output, layer_idx, mode):
        frame, name = mode.split('-')
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
        elif mode in ['hf-down_proj_out']:
            hf_hiddens[layer_idx][name] = output.reshape(-1, hidden_size)
            hf_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['mg-down_proj_out']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, hidden_size)
            mg_hiddens[layer_idx][name + '_weight'] = module.weight

    hfmodel.lm_head.register_forward_hook(partial(print_output_hook, layer_idx=mgargs.num_layers - 1, mode='hf-lmhead'),
                                          with_kwargs=True)

    mgmodel.output_layer.register_forward_hook(
        partial(print_output_hook, layer_idx=mgargs.num_layers - 1, mode='mg-lmhead'), with_kwargs=True)

    for idx, layer in enumerate(hfmodel.model.layers):

        layer.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='hf-layer_in'), with_kwargs=True)

        layer.self_attn.o_proj.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='hf-o_proj_in'),
                                                         with_kwargs=True)

        layer.self_attn.o_proj.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='hf-o_proj_out'),
                                                     with_kwargs=True)

        layer.self_attn.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='hf-attn_out'),
                                              with_kwargs=True)

        layer.mlp.down_proj.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode='hf-down_proj_in'), with_kwargs=True)

        layer.mlp.down_proj.register_forward_hook(
            partial(print_output_hook, layer_idx=idx, mode='hf-down_proj_out'), with_kwargs=True)

    for idx, layer in enumerate(mgmodel.decoder.layers):

        layer.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='mg-layer_in'), with_kwargs=True)

        layer.self_attention.linear_proj.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode='mg-o_proj_in'), with_kwargs=True)

        layer.self_attention.linear_proj.register_forward_hook(
            partial(print_output_hook, layer_idx=idx, mode='mg-o_proj_out'), with_kwargs=True)

        layer.self_attention.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='mg-attn_out'),
                                                   with_kwargs=True)

        layer.mlp.linear_fc2.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode='mg-down_proj_in'), with_kwargs=True)

        layer.mlp.linear_fc2.register_forward_hook(
            partial(print_output_hook, layer_idx=idx, mode='mg-down_proj_out'), with_kwargs=True)


    input_ids = torch.tensor([[1, 2, 3]]).long().cuda()
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(input_ids, -100, True, True, True)
    print(hfmodel)
    print(mgmodel)
    with torch.inference_mode():
        try:
            hfmodel.cuda()
            hfmodel(input_ids=input_ids)
        except torch.cuda.OutOfMemoryError:
            print('oom for huggingface model forward')
        hfmodel.cpu()
        del hfmodel

    with torch.inference_mode():
        try:
            mgmodel.cuda()
            mgmodel(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        except torch.cuda.OutOfMemoryError:
            print('oom for megatron model forward')
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


def add_extra_args(parser):
    parser = get_patch_args(parser)
    parser = add_model_args(parser)
    return parser

def main():
    initialize_megatron(extra_args_provider=add_extra_args)
    args = get_args()

    if args.convert_checkpoint_from_megatron_to_transformers:
        hf_model = AutoModelForCausalLM.from_pretrained(args.hf_ckpt_path)
        mg_model = load_megatron_model(args)
        convert_checkpoint_from_megatron_to_transformers(mg_model, hf_model, args)
        save_hfmodel(args, hf_model)
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(args.load)
        mg_model = model_provider()
        convert_checkpoint_from_transformers_to_megatron(hf_model, mg_model, args)
        check_hf_mg_forward(hf_model, mg_model, args)
        save_mgmodel(mg_model, args)

if __name__ == "__main__":
    main()

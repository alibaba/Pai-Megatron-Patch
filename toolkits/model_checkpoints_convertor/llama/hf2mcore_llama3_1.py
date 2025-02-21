import copy
import json
import os
import re
import sys
from collections import defaultdict
from functools import partial

import safetensors
import torch
from megatron.training import get_args
from megatron.training.checkpointing import (
    get_checkpoint_name,
    get_checkpoint_tracker_filename,
    read_metadata,
)
from megatron.training.initialize import initialize_megatron
from megatron.training.utils import get_ltor_masks_and_position_ids
from transformers import AutoModelForCausalLM

path_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
sys.path.append(os.path.join(path_dir, "examples"))
from llama3_1.pretrain_llama import model_provider
from megatron_patch.arguments import get_patch_args
from toolkits.model_checkpoints_convertor.utils import (
    save_state_dict,
    save_hfmodel
)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

def add_model_args(parser):

    parser.add_argument("--target-tensor-model-parallel-size", type=int, default=1)

    parser.add_argument("--target-pipeline-model-parallel-size", type=int, default=1)

    parser.add_argument(
        "--save-safetensors",
        action="store_true",
    )

    parser.add_argument(
        "--no-rotary-embed-copy",
        action="store_true",
    )

    parser.add_argument(
        "--check-alignment",
        action="store_true",
        help=(
            "check if the model/converted model strictly equivalent with sample input. "
            "Should install flash-attention 2 to avoid numerical losses."
        ),
    )

    parser.add_argument(
        "--check-only",
        default=False,
        action="store_true",
        help=("Load and check if the two checkpoints strictly equivalent."),
    )

    parser.add_argument(
        "--naive-check",
        default=False,
        action="store_true",
        help=("Load and check if the two checkpoints strictly equivalent."),
    )

    parser.add_argument(
        "--max-shard-size",
        type=str,
        default="10GB",
        help=(
            "The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size "
            "lower than this size. If expressed as a string, needs to be digits followed by a unit (like `5MB`). "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )

    parser.add_argument("--hf-ckpt-path", type=str)

    parser.add_argument("--save-path", type=str)

    return parser


def load_megatron_model(args):
    """
    Load a TP1PP1 model(full model) from arbitrary tp-pp rank
    """
    os.makedirs(args.save, exist_ok=True)
    os.system("cp -rf " + args.hf_ckpt_path + "/config*.json " + args.save)
    os.system("cp -rf " + args.hf_ckpt_path + "/tokenizer* " + args.save)
    os.system("cp -rf " + args.hf_ckpt_path + "/vocab.json " + args.save)
    os.system("cp -rf " + args.hf_ckpt_path + "/merges.txt " + args.save)

    os.system("cp -rf " + args.hf_ckpt_path + "/config*.json " + args.load)
    os.system("cp -rf " + args.hf_ckpt_path + "/tokenizer* " + args.load)
    os.system("cp -rf " + args.hf_ckpt_path + "/vocab.json " + args.load)
    os.system("cp -rf " + args.hf_ckpt_path + "/merges.txt " + args.load)
    
    model = model_provider().cpu()
    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size

    model_path = args.load
    tracker_filename = get_checkpoint_tracker_filename(model_path)
    iteration, release = read_metadata(tracker_filename)

    if args.tensor_model_parallel_size > 1:
        args.sequence_parallel = True

    assert args.num_query_groups >= args.target_tensor_model_parallel_size

    head_dim = args.hidden_size // args.num_attention_heads
    group_per_split = args.num_query_groups // args.target_tensor_model_parallel_size

    state_dict = {}
    mid_state = defaultdict(list)

    if args.tensor_model_parallel_size == 1 and args.pipeline_model_parallel_size == 1:
        checkpoint_name = get_checkpoint_name(
            model_path, iteration, release, None, None, None, None, None
        )
        state_dict = torch.load(checkpoint_name, weights_only=False)["model"]
    elif args.tensor_model_parallel_size > 1 and args.pipeline_model_parallel_size == 1:
        for tp_rank in range(args.tensor_model_parallel_size):
            checkpoint_name = get_checkpoint_name(
                model_path, iteration, release, None, tp_rank, None, None, None
            )
            print(f"load {checkpoint_name}")
            split_state = torch.load(checkpoint_name, map_location="cpu", weights_only=False)["model"]
            for k, v in split_state.items():
                mid_state[k].append(v)
        for k, v in mid_state.items():

            if not isinstance(v[0], torch.Tensor) or "norm" in k:
                target_v = v[0]
            elif 'extra_state' in k:
                target_v = None
            elif "embedding" in k or "output_layer" in k:
                target_v = torch.cat(v, dim=0)
            elif "linear_proj" in k or "linear_fc2" in k:
                target_v = torch.cat(v, dim=1)
            elif "linear_qkv.weight" in k:
                viewed = [
                    x.view(group_per_split, -1, head_dim, args.hidden_size)
                    for x in v
                ]
                target_v = torch.cat(viewed, dim=0).view(-1, args.hidden_size)
            elif "linear_qkv.bias" in k:
                viewed = [x.view(group_per_split, -1) for x in v]
                target_v = torch.cat(viewed, dim=0).view(-1)
            elif "linear_fc1" in k:
                viewed = [x.view(2, -1, args.hidden_size) for x in v]
                target_v = torch.cat(viewed, dim=1).view(-1, args.hidden_size)
            else:
                raise ValueError

            state_dict[k] = target_v
    elif args.tensor_model_parallel_size > 1 and args.pipeline_model_parallel_size > 1:
        num_layers = args.num_layers // args.pipeline_model_parallel_size
        layers_to_copy = {}
        for tp_rank in range(args.tensor_model_parallel_size):
            for pp_rank in range(args.pipeline_model_parallel_size):
                layer_offset = pp_rank * num_layers
                for layer in range(num_layers):
                    pp_layer_id = layer + layer_offset
                    layers_to_copy[f"decoder.layers.{layer}"] = pp_layer_id
                checkpoint_name = get_checkpoint_name(
                    model_path, iteration, release, True, tp_rank, pp_rank, None, None
                )
                print(f"load {checkpoint_name}")
                split_state = torch.load(checkpoint_name, map_location="cpu", weights_only=False)["model"]
                for k, v in split_state.items():
                    try:
                        pattern = re.compile(r"\d+")
                        res = pattern.findall(k)
                        k = re.sub(
                            r"decoder.layers.\d+",
                            "decoder.layers."
                            + str(layers_to_copy["decoder.layers." + res[0]]),
                            k,
                        )
                        mid_state[k].append(v)
                    except:
                        mid_state[k].append(v)
        for k, v in mid_state.items():
            if not isinstance(v[0], torch.Tensor) or "norm" in k:
                target_v = v[0]
            elif 'extra_state' in k:
                target_v = None
            elif "embedding" in k or "output_layer" in k:
                target_v = torch.cat(v, dim=0)
            elif "linear_proj" in k or "linear_fc2" in k:
                target_v = torch.cat(v, dim=1)
            elif "linear_qkv.weight" in k:
                viewed = [
                    x.view(group_per_split, -1, head_dim, args.hidden_size)
                    for x in v
                ]
                target_v = torch.cat(viewed, dim=0).view(-1, args.hidden_size)
            elif "linear_qkv.bias" in k:
                viewed = [x.view(group_per_split, -1) for x in v]
                target_v = torch.cat(viewed, dim=0).view(-1)
            elif "linear_fc1" in k:
                viewed = [x.view(2, -1, args.hidden_size) for x in v]
                target_v = torch.cat(viewed, dim=1).view(-1, args.hidden_size)
            else:
                raise ValueError
            state_dict[k] = target_v
    incompat_keys = model.load_state_dict(state_dict, strict=False)

    unexpected_keys = []
    for key in incompat_keys.unexpected_keys:
        if "extra_state" not in key:
            unexpected_keys.append(key)
    assert len(unexpected_keys) == 0, "Unexpected Keys: " + str(unexpected_keys)
    missed_keys = []
    for key in incompat_keys.missing_keys:
        if "extra_state" not in key:
            missed_keys.append(key)
    assert len(missed_keys) == 0, "Missing Keys: " + str(missed_keys)
    return model


def add_extra_args(parser):
    parser = get_patch_args(parser)
    parser = add_model_args(parser)
    return parser


def check_layer(layers_to_copy, k):
    pattern = re.compile(r"decoder.layers.\d+")
    res = pattern.findall(k)
    return res and res[0] in layers_to_copy.keys()


def naive_check_forward(model1, model2, args):
    if args.fp16:
        model1 = model1.half()
        model2 = model2.half()
    elif args.bf16:
        model1 = model1.bfloat16()
        model2 = model2.bfloat16()

    input_ids = torch.tensor([[1, 2, 3]]).long().cuda()
    is_oom = False
    with torch.inference_mode():
        try:
            model1.cuda()
            hflogits = model1(input_ids=input_ids).logits
        except torch.cuda.OutOfMemoryError:
            print("oom for model1 forward")
            is_oom = True
        model1.cpu()
        del model1

    with torch.inference_mode():
        try:
            model2.cuda()
            mglogits = model2(
                input_ids=input_ids,
            ).logits
        except torch.cuda.OutOfMemoryError:
            print("oom for model2 forward")
            is_oom = True
        del model2
    epsilon = 1e-5
    if not is_oom:
        same_num = (hflogits != mglogits).sum()
        diff_num = ((hflogits - mglogits) > epsilon).sum()
        diff_max = (hflogits - mglogits).abs().max()
        print(
            f"logits: {same_num}, diff>{epsilon}:[{diff_num}/{hflogits.numel()}] diff_max:{diff_max}"
        )


def check_hf_mg_forward(hfmodel, mgmodel, mgargs):
    if mgargs.fp16:
        mgmodel = mgmodel.half()
        hfmodel = hfmodel.half()
    elif mgargs.bf16:
        mgmodel = mgmodel.bfloat16()
        hfmodel = hfmodel.bfloat16()
    hf_hiddens = [{} for _ in range(mgargs.num_layers)]
    mg_hiddens = [{} for _ in range(mgargs.num_layers)]

    hidden_size = mgargs.hidden_size
    vocab_size = mgargs.padded_vocab_size

    def print_input_hook(module, args, kwargs, layer_idx, mode):
        frame, name = mode.split("-")
        if frame == "hf" and "attn_in" in mode:
            hf_hiddens[layer_idx][name] = kwargs.get("hidden_states")[0]
        elif frame == "hf":
            hf_hiddens[layer_idx][name] = args[0].transpose(0, 1)
        elif frame == "mg" and "layer" in mode:
            mg_hiddens[layer_idx][name] = kwargs.get("hidden_states")
        elif frame == "mg" and mode == "mg-attn_in":
            mg_hiddens[layer_idx][name] = args[0][:, 0]
        elif frame == "mg":
            mg_hiddens[layer_idx][name] = args[0]

    def print_output_hook(module, args, kwargs, output, layer_idx, mode):
        frame, name = mode.split("-")
        if mode in ["hf-lmhead"]:
            hf_hiddens[layer_idx][name] = output.transpose(0, 1).reshape(-1, vocab_size)
            hf_hiddens[layer_idx][name + "_weight"] = module.weight
            hf_hiddens[layer_idx][name + "_token"] = output.transpose(0, 1).max(dim=-1)[
                1
            ]
        elif mode in ["mg-lmhead"]:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, vocab_size)
            mg_hiddens[layer_idx][name + "_weight"] = module.weight
            mg_hiddens[layer_idx][name + "_token"] = output[0].max(dim=-1)[1]
        elif mode in ["hf-o_proj_out"]:
            hf_hiddens[layer_idx][name] = output
            hf_hiddens[layer_idx][name + "_weight"] = module.weight
        elif mode in ["mg-o_proj_out"]:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, hidden_size)
            mg_hiddens[layer_idx][name + "_weight"] = module.weight
        elif mode in ["hf-attn_out"]:
            hf_hiddens[layer_idx][name] = output[0].reshape(-1, hidden_size)
        elif mode in ["hf-core_attn_out"]:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, hidden_size)
        elif mode in ["mg-core_attn_out"]:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, hidden_size)
        elif mode in ["mg-attn_out"]:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, hidden_size)
        elif mode in ["mg-mlp_out"]:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, hidden_size)
        elif mode in ["hf-mlp_out"]:
            hf_hiddens[layer_idx][name] = output[0].reshape(-1, hidden_size)

    hfmodel.lm_head.register_forward_hook(
        partial(print_output_hook, layer_idx=mgargs.num_layers - 1, mode="hf-lmhead"),
        with_kwargs=True,
    )

    mgmodel.output_layer.register_forward_hook(
        partial(print_output_hook, layer_idx=mgargs.num_layers - 1, mode="mg-lmhead"),
        with_kwargs=True,
    )

    for idx, layer in enumerate(hfmodel.model.layers):

        layer.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode="hf-layer_in"),
            with_kwargs=True,
        )

        layer.self_attn.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode="hf-attn_in"),
            with_kwargs=True,
        )

        layer.self_attn.o_proj.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode="hf-o_proj_in"),
            with_kwargs=True,
        )

        layer.self_attn.o_proj.register_forward_hook(
            partial(print_output_hook, layer_idx=idx, mode="hf-o_proj_out"),
            with_kwargs=True,
        )

        layer.self_attn.register_forward_hook(
            partial(print_output_hook, layer_idx=idx, mode="hf-attn_out"),
            with_kwargs=True,
        )

        layer.post_attention_layernorm.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode="hf-pre_ln_in"),
            with_kwargs=True,
        )

        layer.mlp.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode="hf-mlp_in"),
            with_kwargs=True,
        )

        layer.mlp.down_proj.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode="hf-down_in"),
            with_kwargs=True,
        )

        layer.mlp.register_forward_hook(
            partial(print_output_hook, layer_idx=idx, mode="hf-mlp_out"),
            with_kwargs=True,
        )

    for idx, layer in enumerate(mgmodel.decoder.layers):

        layer.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode="mg-layer_in"),
            with_kwargs=True,
        )

        layer.self_attention.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode="mg-attn_in"),
            with_kwargs=True,
        )

        layer.self_attention.linear_proj.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode="mg-o_proj_in"),
            with_kwargs=True,
        )

        layer.self_attention.linear_proj.register_forward_hook(
            partial(print_output_hook, layer_idx=idx, mode="mg-o_proj_out"),
            with_kwargs=True,
        )

        layer.self_attention.register_forward_hook(
            partial(print_output_hook, layer_idx=idx, mode="mg-attn_out"),
            with_kwargs=True,
        )

        layer.pre_mlp_layernorm.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode="mg-pre_ln_in"),
            with_kwargs=True,
        )

        layer.mlp.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode="mg-mlp_in"),
            with_kwargs=True,
        )

        layer.mlp.linear_fc2.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode="mg-down_in"),
            with_kwargs=True,
        )

        layer.mlp.register_forward_hook(
            partial(print_output_hook, layer_idx=idx, mode="mg-mlp_out"),
            with_kwargs=True,
        )

    input_ids = torch.tensor([[1, 2, 3]]).long().cuda()
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        input_ids, -100, True, True, True
    )
    print(hfmodel)
    print(mgmodel)
    is_oom = False
    with torch.inference_mode():
        try:
            hfmodel.cuda()
            hflogits = hfmodel(input_ids=input_ids).logits
        except torch.cuda.OutOfMemoryError:
            print("oom for huggingface model forward")
            is_oom = True
        hfmodel.cpu()
        del hfmodel

    with torch.inference_mode():
        try:
            mgmodel.cuda()
            mglogits = mgmodel(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        except torch.cuda.OutOfMemoryError:
            print("oom for megatron model forward")
            is_oom = True
        del mgmodel

    epsilon = 1e-5
    for idx, (hfh, mgh) in enumerate(zip(hf_hiddens, mg_hiddens)):
        assert len(hfh) == len(mgh)
        for k, hfv in hfh.items():
            mgv, hfv = mgh[k].cpu(), hfv.cpu()
            same_num = (hfv != mgv).sum()
            diff_num = ((hfv - mgv) > epsilon).sum()
            diff_max = (hfv - mgv).abs().max()
            print(
                f"layer:{idx}, {k}, diff: {same_num}, diff>{epsilon}:[{diff_num}/{hfv.numel()}] diff_max:{diff_max}"
            )

    if not is_oom:
        same_num = (hflogits != mglogits).sum()
        diff_num = ((hflogits - mglogits) > epsilon).sum()
        diff_max = (hflogits - mglogits).abs().max()
        print(
            f"logits: {same_num}, diff>{epsilon}:[{diff_num}/{hflogits.numel()}] diff_max:{diff_max}"
        )


def _remap_no_hf_impl_values(
    k,
    v,
    num_query_groups,
    head_dim,
    hidden_size,
    group_per_split,
    tp_rank,
    tensor_model_parallel_size,
    ffn_hidden_size,
):
    if not isinstance(v, torch.Tensor):
        target_v = v
    elif "linear_qkv.weight" in k:
        viewed = v.view(num_query_groups, -1, head_dim, hidden_size)
        viewed = viewed[group_per_split * tp_rank : group_per_split * (tp_rank + 1)]
        target_v = viewed.view(-1, hidden_size)
    elif "linear_qkv.bias" in k:
        viewed = v.view(num_query_groups, -1, head_dim)
        viewed = viewed[group_per_split * tp_rank : group_per_split * (tp_rank + 1)]
        target_v = viewed.view(-1)
    elif "linear_proj" in k or "linear_fc2" in k:
        seg = v.shape[1] // tensor_model_parallel_size
        target_v = v[:, seg * tp_rank : seg * (tp_rank + 1)]
    elif "embedding" in k or "output_layer" in k:
        seg = v.shape[0] // tensor_model_parallel_size
        target_v = v[seg * tp_rank : seg * (tp_rank + 1)]
    elif "linear_fc1" in k and "norm" not in k:
        viewed = v.view(-1, ffn_hidden_size, hidden_size)
        seg = ffn_hidden_size // tensor_model_parallel_size
        target_v = viewed[:, seg * tp_rank : seg * (tp_rank + 1), :].reshape(
            -1, hidden_size
        )
    else:
        target_v = v
    return target_v

def save_mgmodel(mgmodel, args):
    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size

    os.makedirs(args.save, exist_ok=True)
    os.system("cp -rf " + args.load + "/config*.json " + args.save)
    os.system("cp -rf " + args.load + "/tokenizer* " + args.save)

    head_dim = args.hidden_size // args.num_attention_heads
    group_per_split = args.num_query_groups // args.target_tensor_model_parallel_size
    tracker_filepath = os.path.join(args.save, "latest_checkpointed_iteration.txt")
    with open(tracker_filepath, "w") as f:
        f.write("release")

    full_model = mgmodel.state_dict_for_save_checkpoint()
    not_processed = {}

    num_layers = args.num_layers // args.pipeline_model_parallel_size
    for k in list(full_model.keys()):
        if 'extra_state' in k:
            # NOTE: since TE 1.14, fp8 metadata will be saved as tensor. 
            # Always drop these values in the MG ckpt to avoid potential issue.
            # This should work fine because fp8 metadata is not supported by HF ckpt.
            full_model[k] = None
        elif full_model[k] is None:
            full_model.pop(k)

    if args.tensor_model_parallel_size == 1 and args.pipeline_model_parallel_size == 1:
        checkpoint_name = get_checkpoint_name(args.save, 0, True)
        save_state_dict(args, [full_model], checkpoint_name)
    elif args.tensor_model_parallel_size > 1 and args.pipeline_model_parallel_size == 1:
        for tp_rank in range(args.tensor_model_parallel_size):
            model_split = {}
            checkpoint_name = get_checkpoint_name(args.save, 0, True, None, tp_rank)
            print(f"tensor_parallel, save model to {checkpoint_name}")
            for k, v in full_model.items():

                model_split[k] = _remap_no_hf_impl_values(
                    k,
                    v,
                    args.num_query_groups,
                    head_dim,
                    args.hidden_size,
                    group_per_split,
                    tp_rank,
                    args.tensor_model_parallel_size,
                    args.ffn_hidden_size,
                )

            save_state_dict(args, [model_split], checkpoint_name)
    else:
        for tp_rank in range(args.tensor_model_parallel_size):
            for pp_rank in range(args.pipeline_model_parallel_size):
                model_split = {}
                layer_offset = pp_rank * num_layers
                layers_to_copy = {}
                for layer in range(num_layers):
                    pp_layer_id = layer + layer_offset
                    layers_to_copy[f"decoder.layers.{pp_layer_id}"] = layer

                checkpoint_name = get_checkpoint_name(
                    args.save, 0, True, True, tp_rank, pp_rank
                )
                print(
                    f"tensor_parallel & pipeline_parallel, save model to {checkpoint_name}"
                )

                for k, v in full_model.items():
                    if check_layer(layers_to_copy, k):
                        pattern = re.compile(r"\d+")
                        res = pattern.findall(k)
                        k = re.sub(
                            r"decoder.layers.\d+",
                            "decoder.layers."
                            + str(layers_to_copy["decoder.layers." + res[0]]),
                            k,
                        )
                    elif not (
                        "word_embeddings" in k
                        or "output_layer" in k
                        or "final_layernorm" in k
                    ):
                        continue


                    target_v = _remap_no_hf_impl_values(
                        k,
                        v,
                        args.num_query_groups,
                        head_dim,
                        args.hidden_size,
                        group_per_split,
                        tp_rank,
                        args.tensor_model_parallel_size,
                        args.ffn_hidden_size,
                    )

                    if "word_embeddings" in k:
                        if pp_rank == 0:
                            model_split[k] = target_v
                    elif "output_layer" in k or "final_layernorm" in k:
                        if pp_rank == args.pipeline_model_parallel_size - 1:
                            model_split[k] = target_v
                    else:
                        model_split[k] = target_v
                save_state_dict(args, [model_split], checkpoint_name)
    print(f"megatron model is save to {args.save}")
    return not_processed.keys()


def convert_checkpoint_from_megatron_to_transformers(mgmodel, hfmodel, args):
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

    with torch.no_grad():
        # 1. embedding, hfmodel only have word_embeddings
        hfmodel.model.embed_tokens.weight.copy_(
            mgmodel.embedding.word_embeddings.weight
        )

        # 2. rotary embeddings, can be skipped with option `--no-rotary-embed-copy`
        if not args.no_rotary_embed_copy:
            hfmodel.model.rotary_emb.inv_freq.copy_(mgmodel.rotary_pos_emb.inv_freq)

        # 3. all decode layers
        for mglayer, hflayer in zip(mgmodel.decoder.layers, hfmodel.model.layers):
            # 3.1. input layernorm (RMSNorm, no bias)
            if use_te:
                hflayer.input_layernorm.weight.copy_(
                    mglayer.self_attention.linear_qkv.layer_norm_weight
                )
            else:
                hflayer.input_layernorm.weight.copy_(mglayer.input_layernorm.weight)

            # 3.2 linear qkv, no bias
            qkv_weight = mglayer.self_attention.linear_qkv.weight.view(
                num_query_groups, -1, head_dim, hidden_size
            )
            q_weight, k_weight, v_weight = torch.split(
                qkv_weight,
                split_size_or_sections=[value_num_per_group, 1, 1],
                dim=1,
            )
            hflayer.self_attn.q_proj.weight.copy_(q_weight.reshape(-1, hidden_size))
            hflayer.self_attn.k_proj.weight.copy_(k_weight.reshape(-1, hidden_size))
            hflayer.self_attn.v_proj.weight.copy_(v_weight.reshape(-1, hidden_size))


            # 3.3 linear proj
            hflayer.self_attn.o_proj.weight.copy_(
                mglayer.self_attention.linear_proj.weight
            )

            # 3.5 MLP:
            gate_weight, fc1_weight = torch.split(
                mglayer.mlp.linear_fc1.weight,
                split_size_or_sections=args.ffn_hidden_size,
            )
            hflayer.mlp.gate_proj.weight.copy_(gate_weight)
            hflayer.mlp.up_proj.weight.copy_(fc1_weight)
            hflayer.mlp.down_proj.weight.copy_(mglayer.mlp.linear_fc2.weight)

            if use_te:
                hflayer.post_attention_layernorm.weight.copy_(
                    mglayer.mlp.linear_fc1.layer_norm_weight
                )
            else:
                hflayer.post_attention_layernorm.weight.copy_(
                    mglayer.pre_mlp_layernorm.weight
                )

        hfmodel.model.norm.weight.copy_(mgmodel.final_layernorm.weight)
        hfmodel.lm_head.weight.copy_(mgmodel.output_layer.weight)


def convert_checkpoint_from_transformers_to_megatron(hfmodel, mgmodel, args):

    if args.fp16:
        mgmodel = mgmodel.half()
        hfmodel = hfmodel.half()
    elif args.bf16:
        mgmodel = mgmodel.bfloat16()
        hfmodel = hfmodel.bfloat16()

    assert args.num_query_groups >= args.target_tensor_model_parallel_size
    num_attention_heads = args.num_attention_heads
    num_query_groups = args.num_query_groups
    hidden_size = args.hidden_size
    head_dim = hidden_size // num_attention_heads
    use_te = args.transformer_impl == "transformer_engine"

    with torch.no_grad():
        # 1. embedding, hfmodel only have word_embeddings
        mgmodel.embedding.word_embeddings.weight.copy_(
            hfmodel.model.embed_tokens.weight
        )
        # 2. rotary embeddings, can be skipped with option `--no-rotary-embed-copy`
        if not args.no_rotary_embed_copy:
            mgmodel.rotary_pos_emb.inv_freq.copy_(hfmodel.model.rotary_emb.inv_freq)

        # 3. all decode layers
        for mglayer, hflayer in zip(mgmodel.decoder.layers, hfmodel.model.layers):
            # 3.1. input layernorm (RMSNorm, no bias)
            if use_te:
                mglayer.self_attention.linear_qkv.layer_norm_weight.copy_(
                    hflayer.input_layernorm.weight
                )
            else:
                mglayer.input_layernorm.weight.copy_(hflayer.input_layernorm.weight)

            # 3.2 linear qkv, no bias
            q_proj_weight = hflayer.self_attn.q_proj.weight.view(
                num_query_groups, -1, head_dim, hidden_size
            )
            k_proj_weight = hflayer.self_attn.k_proj.weight.view(
                num_query_groups, -1, head_dim, hidden_size
            )
            v_proj_weight = hflayer.self_attn.v_proj.weight.view(
                num_query_groups, -1, head_dim, hidden_size
            )
            qkv_proj = (
                torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=1)
                .view(-1, hidden_size)
                .contiguous()
            )
            mglayer.self_attention.linear_qkv.weight.copy_(qkv_proj)


            # 3.3 linear proj
            mglayer.self_attention.linear_proj.weight.copy_(
                hflayer.self_attn.o_proj.weight
            )

            # NOTE: construct mlp first.
            # 3.5 MLP:
            fc1_weight = torch.cat(
                [hflayer.mlp.gate_proj.weight, hflayer.mlp.up_proj.weight]
            )
            mglayer.mlp.linear_fc1.weight.copy_(fc1_weight)
            mglayer.mlp.linear_fc2.weight.copy_(hflayer.mlp.down_proj.weight)


            # 3.4 post-attn / pre-mlp layernorm, no bias
            if use_te:
                mglayer.mlp.linear_fc1.layer_norm_weight.copy_(
                    hflayer.post_attention_layernorm.weight
                )
            else:
                mglayer.pre_mlp_layernorm.weight.copy_(
                    hflayer.post_attention_layernorm.weight
                )

        # 4. final layernorm
        # NOTE: we move the final layernorm out of decoder to apply LLaMARMSNorm
        # mgmodel.decoder.final_layernorm.weight.copy_(hfmodel.model.norm.weight)
        mgmodel.final_layernorm.weight.copy_(hfmodel.model.norm.weight)
        # 5. output layer
        mgmodel.output_layer.weight.copy_(hfmodel.lm_head.weight)


def main():
    initialize_megatron(extra_args_provider=add_extra_args)
    args = get_args()
    load_args = {}
    

    if args.naive_check:
        # check two HF model's output
        # Load hf1 <- args.load and hf2 <- args.hf-ckpt-path
        load_args.update({"pretrained_model_name_or_path": args.load})
        model1 = AutoModelForCausalLM.from_pretrained(**load_args).cpu()
        load_args.update({"pretrained_model_name_or_path": args.hf_ckpt_path})
        model2 = AutoModelForCausalLM.from_pretrained(**load_args).cpu()
        torch.cuda.empty_cache()
        naive_check_forward(model1, model2, args)
        return

    if args.check_alignment and args.check_only:
        # Load two models: hf <- args.hf-ckpt-path mg <- args.load
        # NOTE: args.hf-ckpt-path is the path to converted checkpoint
        load_args.update({"pretrained_model_name_or_path": args.hf_ckpt_path})
        print("Initialize HuggingFace model with:", load_args, flush=True)
        hf_model = AutoModelForCausalLM.from_pretrained(**load_args).cpu()
        mg_model = load_megatron_model(args).cpu()
        check_hf_mg_forward(hf_model, mg_model, args)
        return

    if args.convert_checkpoint_from_megatron_to_transformers:
        # Load one model: mg <- args.load and initialize one model: hf <- args.hf-ckpt-path
        # NOTE: args.hf-ckpt-path is the path to official checkpoint
        load_args.update({"pretrained_model_name_or_path": args.hf_ckpt_path})
        print("Initialize HuggingFace model with:", load_args, flush=True)
        hf_model = AutoModelForCausalLM.from_pretrained(**load_args).cpu()
        mg_model = load_megatron_model(args).cpu()
        convert_checkpoint_from_megatron_to_transformers(mg_model, hf_model, args)
    else:
        # NOTE: args.load is the path to the hf checkpoint
        load_args.update({"pretrained_model_name_or_path": args.load})
        print("Load HuggingFace model with:", load_args, flush=True)
        hf_model = AutoModelForCausalLM.from_pretrained(**load_args).cpu()
        mg_model = model_provider().cpu()
        convert_checkpoint_from_transformers_to_megatron(hf_model, mg_model, args)

    if args.check_alignment:
        check_hf_mg_forward(hf_model, mg_model, args)

    if args.convert_checkpoint_from_megatron_to_transformers:
        save_hfmodel(args, hf_model)
    else:
        save_mgmodel(mg_model, args)


if __name__ == "__main__":
    main()

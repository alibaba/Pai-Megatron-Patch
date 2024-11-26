import copy
import json
import os
import re
import sys
import clip
from collections import defaultdict
import safetensors
import torch

from megatron.training import get_args
from megatron.training.checkpointing import (
    get_checkpoint_name,
    get_checkpoint_tracker_filename,
    read_metadata,
)
from megatron.training.initialize import initialize_megatron
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import (
    shard_checkpoint,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)

path_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
sys.path.append(os.path.join(path_dir, "examples"))
from mistral.pretrain_mcore_mistral import model_provider
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

    parser.add_argument("--clip-ckpt-path", type=str)


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
        state_dict = torch.load(checkpoint_name)["model"]
    elif args.tensor_model_parallel_size > 1 and args.pipeline_model_parallel_size == 1:
        for tp_rank in range(args.tensor_model_parallel_size):
            checkpoint_name = get_checkpoint_name(
                model_path, iteration, release, None, tp_rank, None, None, None
            )
            print(f"load {checkpoint_name}")
            split_state = torch.load(checkpoint_name, map_location="cpu")["model"]
            for k, v in split_state.items():
                mid_state[k].append(v)
        for k, v in mid_state.items():

            if not isinstance(v[0], torch.Tensor) or "norm" in k:
                target_v = v[0]
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
                split_state = torch.load(checkpoint_name, map_location="cpu")["model"]
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


def save_state_dict(args, model, checkpoint_name):
    state_dict = {}
    state_dict["args"] = args
    state_dict["checkpoint_version"] = 3.0
    state_dict["iteration"] = 0
    state_dict["model"] = model
    os.makedirs(os.path.dirname(checkpoint_name), exist_ok=True)
    print(f"save model part {checkpoint_name}")
    torch.save(clone_state_dict(state_dict), checkpoint_name)


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
        if full_model[k] is None or "_extra_state" in k:
            v = full_model.pop(k)
            if v is not None:
                not_processed[k] = v

    if args.tensor_model_parallel_size == 1 and args.pipeline_model_parallel_size == 1:
        checkpoint_name = get_checkpoint_name(args.save, 0, True)
        save_state_dict(args, full_model, checkpoint_name)
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

            save_state_dict(args, model_split, checkpoint_name)
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
                save_state_dict(args, model_split, checkpoint_name)
    print(f"megatron model is save to {args.save}")
    return not_processed.keys()


def save_hfmodel(args, model):
    output_state_dict = model.state_dict()
    max_shard_size = args.max_shard_size
    shards, index = shard_checkpoint(output_state_dict, max_shard_size=max_shard_size)
    os.makedirs(args.save, exist_ok=True)
    for shard_file, shard in shards.items():
        if args.save_safetensors:
            shard_file = shard_file.replace("pytorch_", "")
            shard_file = shard_file.replace(".bin", ".safetensors")
            target_file = os.path.join(args.save, shard_file)
            print(f"huggingface model is save to {target_file}")
            new_shard = {}
            for k, v in shard.items():
                new_shard[k] = copy.deepcopy(v)
            safetensors.torch.save_file(
                clone_state_dict(new_shard), target_file, metadata={"format": "pt"}
            )
        else:
            target_file = os.path.join(args.save, shard_file)
            print(f"huggingface model is save to {target_file}")
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
        mgmodel.decoder.final_layernorm.weight.copy_(hfmodel.model.norm.weight)
        # 5. output layer
        mgmodel.output_layer.weight.copy_(hfmodel.lm_head.weight)

def convert_clip(download_root, output_path, tensor_parallel_size, use_te):
    device = "cuda"

    model, _ = clip.load("ViT-L/14@336px", device=device, download_root=download_root)

    state_dict = model.state_dict()
    new_state_dicts = [{"model": dict()} for _ in range(tensor_parallel_size)]

    # Indices from mapping pytorch multihead attention to megatron.
    kv_channels = 64
    hidden_dim = 1024
    num_heads = 16
    indices = []
    for i in range(num_heads):
        lb = i * kv_channels
        ub = (i + 1) * kv_channels
        indices.append(torch.arange(lb, ub, dtype=torch.int))
        indices.append(torch.arange(hidden_dim + lb, hidden_dim + ub, dtype=torch.int))
        indices.append(torch.arange(2 * hidden_dim + lb, 2 * hidden_dim + ub, dtype=torch.int))

    indices = torch.cat(indices)

    for name, tensor in state_dict.items():
        # Skip text model.
        if "visual" not in name:
            continue

        # Skip final layers not used in our model.
        if name == "visual.proj" or "ln_post" in name:
            continue

        # Map parameter names to ones used in megatron.
        new_name = ""
        new_tensor = tensor
        if new_tensor.dtype == torch.float16:
            new_tensor = new_tensor.to(torch.float32)

        # This is used for chunking some tensors to target tensor parallel size.
        chunk_dim = None

        if "class_embedding" in name:
            new_name = "class_token"
            # Our model uses class token that is expanded to input dimensions already.
            new_tensor = new_tensor.expand(1, 1, -1)
        elif "positional_embedding" in name:
            new_name = "position_embeddings.weight"
        elif "conv1" in name:
            new_name = "conv1.weight"
        elif "ln_pre.weight" in name:
            new_name = "ln_pre.weight"
        elif "ln_pre.bias" in name:
            new_name = "ln_pre.bias"
        elif "transformer.resblocks" in name:
            layer_idx = name.split(".")[3]
            base = f"decoder.layers.{layer_idx}"

            if "attn.in_proj_weight" in name:
                new_name = f"{base}.self_attention.linear_qkv.weight"
                new_tensor = new_tensor[indices]
                chunk_dim = 0
            elif "attn.in_proj_bias" in name:
                new_name = f"{base}.self_attention.linear_qkv.bias"
                new_tensor = new_tensor[indices]
                chunk_dim = 0
            elif "attn.out_proj.weight" in name:
                new_name = f"{base}.self_attention.linear_proj.weight"
                chunk_dim = 1
            elif "attn.out_proj.bias" in name:
                new_name = f"{base}.self_attention.linear_proj.bias"
            elif "ln_1.weight" in name:
                new_name = f"{base}.input_layernorm.weight"
                if use_te:
                    new_name = f"{base}.self_attention.linear_qkv.layer_norm_weight"
            elif "ln_1.bias" in name:
                new_name = f"{base}.input_layernorm.bias"
                if use_te:
                    new_name = f"{base}.self_attention.linear_qkv.layer_norm_bias"
            elif "mlp.c_fc.weight" in name:
                new_name = f"{base}.mlp.linear_fc1.weight"
                chunk_dim = 0
            elif "mlp.c_fc.bias" in name:
                new_name = f"{base}.mlp.linear_fc1.bias"
                chunk_dim = 0
            elif "mlp.c_proj.weight" in name:
                new_name = f"{base}.mlp.linear_fc2.weight"
                chunk_dim = 1
            elif "mlp.c_proj.bias" in name:
                new_name = f"{base}.mlp.linear_fc2.bias"
            elif "ln_2.weight" in name:
                new_name = f"{base}.pre_mlp_layernorm.weight"
                if use_te:
                    new_name = f"{base}.mlp.linear_fc1.layer_norm_weight"
            elif "ln_2.bias" in name:
                new_name = f"{base}.pre_mlp_layernorm.bias"
                if use_te:
                    new_name = f"{base}.mlp.linear_fc1.layer_norm_bias"

        assert new_name != "", f"unexpected layer name {name}"

        if chunk_dim is None:
            new_tensors = [new_tensor for _ in range(tensor_parallel_size)]
        else:
            new_tensors = torch.chunk(new_tensor, tensor_parallel_size, dim=chunk_dim)

        for i in range(tensor_parallel_size):
            # chunk() creates a view of a bigger tensor. clone() is used here to avoid excessive storage.
            new_state_dicts[i]["model"][new_name] = new_tensors[i].clone()

            # TE sets _extra_state (for FP8 purposes), so set an empty one here for compatibility.
            extra_state_layers = ("linear_qkv", "linear_proj", "linear_fc1", "linear_fc2")
            is_extra_state_layer = any([l in new_name for l in extra_state_layers])
            if use_te and is_extra_state_layer:
                layer = new_name.split(".")[-2]
                if layer in extra_state_layers:
                    extra_state_name = (
                        new_name[: new_name.rfind(".") + 1] + "_extra_state"
                    )  # Replace the weight name.
                    new_state_dicts[i]["model"][extra_state_name] = None

    for i in range(tensor_parallel_size):
        output_dir_tp = os.path.join(output_path, "clip_release", f"mp_rank_0{i}")
        os.makedirs(output_dir_tp, exist_ok=True)
        output_path_tp = os.path.join(output_dir_tp, "model_optim_rng.pt")
        torch.save(new_state_dicts[i], output_path_tp)


def combine(save_dir, tensor_parallel_size):

    input_files = []
    output_files = []
    for i in range(tensor_parallel_size):
        llm_dir_tp = os.path.join(save_dir, "release", f"mp_rank_0{i}")
        llm_path_tp = os.path.join(llm_dir_tp, "model_optim_rng.pt")
        input_files.append(llm_path_tp)
        clip_dir_tp = os.path.join(save_dir, "clip_release", f"mp_rank_0{i}")
        clip_path_tp = os.path.join(clip_dir_tp, "model_optim_rng.pt")
        input_files.append(clip_path_tp)
        output_dir_tp = os.path.join(save_dir, "iter_0000001", f"mp_rank_0{i}")
        os.makedirs(output_dir_tp, exist_ok=True)
        output_files.append(os.path.join(output_dir_tp, "model_optim_rng.pt"))

    module_prefixes = ["language_model", "vision_model"] * tensor_parallel_size
    num_inputs_per_output = int(len(input_files) / len(output_files))

    for output_idx, output_file in enumerate(output_files):
        combined_state_dict = None

        lb = output_idx * num_inputs_per_output
        ub = (output_idx + 1) * num_inputs_per_output
        current_input_files = input_files[lb:ub]
        current_module_prefixes = module_prefixes[lb:ub]

        for i, (input_file, module_prefix) in enumerate(
            zip(current_input_files, current_module_prefixes)
        ):
            # initialize the combined state dict using the first provided input file
            current_state_dict = torch.load(input_file)
            if i == 0:
                combined_state_dict = current_state_dict.copy()
                combined_state_dict["model"] = dict()

            # copy model state dict and prefix names with the given module keys.
            for k, v in current_state_dict["model"].items():
                combined_state_dict["model"]["%s.%s" % (module_prefix, k)] = v

        torch.save(combined_state_dict, output_file)
        print("combined saved:", output_file)

    tracker_filepath = os.path.join(save_dir, "latest_checkpointed_iteration.txt")
    with open(tracker_filepath, "w") as f:
        f.write("1")

def main():
    initialize_megatron(extra_args_provider=add_extra_args)
    args = get_args()
    load_args = {}

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


    if args.convert_checkpoint_from_megatron_to_transformers:
        save_hfmodel(args, hf_model)
    else:
        save_mgmodel(mg_model, args)

    use_te = args.transformer_impl == "transformer_engine"
    convert_clip(args.clip_ckpt_path, args.save, args.target_tensor_model_parallel_size, use_te)

    combine(args.save, args.target_tensor_model_parallel_size)


if __name__ == "__main__":
    main()

# Copyright (c) 2023 Alibaba PAI and Nvidia Megatron-LM Team.
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

import argparse
import os
import re
import torch
import types
from collections import OrderedDict

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer


def add_args(parser):
    parser.add_argument(
        "--convert_checkpoint_from_megatron_to_transformers",
        action="store_true",
        help=(
            "If True, convert a Megatron checkpoint to a Transformers checkpoint. "
            "If False, convert a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--load_path",
        type=str,
        required=True,
        help="Path to the checkpoint to convert.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to the converted checkpoint.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="model name",
    )

    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help=(
            "world_size"
        ),
    )

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
            "The tensor model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )

    parser.add_argument(
        "--target_params_dtype",
        type=str,
        default="fp32",
        help=(
            "The dtype of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )

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

    parser.add_argument("--print-checkpoint-structure", action="store_true")

    return parser


internal_to_output_mapping = {
    "self_attn.dense": "self_attention.linear_proj",
    "mlp.dense_h_to_4h_1": "mlp.linear_fc1_1",
    "mlp.dense_h_to_4h_2": "mlp.linear_fc1_2",
    "mlp.dense_4h_to_h": "mlp.linear_fc2",
}

megatron_to_transformers = {
    "self_attention.linear_proj": "self_attn.o_proj",
    "mlp.router": "block_sparse_moe.gate",
}

tensor_parallel_params = [
    # megatron-lm layers to merge across tp ranks
    "self_attn.query.weight",
    "self_attn.key_value.weight",
    "self_attn.dense.weight",
    "mlp.dense_h_to_4h_1.weight",
    "mlp.dense_h_to_4h_2.weight",
    "mlp.dense_4h_to_h.weight",
]

tensor_parallel_params_mg = [
    # megatron-lm layers to merge across tp ranks
    'self_attention.linear_proj.weight',
    'self_attention.linear_qkv.weight',
]

column_split_tensor_parallel_params = [
    "self_attn.dense.weight",
    "mlp.dense_4h_to_h.weight",
]

column_split_tensor_parallel_params_mg = [
    'self_attention.linear_proj'
]


def get_checkpoint_sub_dir_name(tp_rank, pp_rank, pp_size, ep_rank, ep_size):
    sub_dir_name = f"mp_rank_{tp_rank:02d}"
    if pp_size > 1: sub_dir_name = f"{sub_dir_name}_{pp_rank:03d}"
    if ep_size > 1: sub_dir_name = f"{sub_dir_name}_{ep_rank:03d}"
    return sub_dir_name


def get_megatron_sharded_states(args, tp_size, pp_size, pp_rank):
    """
    Get sharded checkpoints from NVIDIA Megatron-LM checkpoint based on the provided tensor parallel size, pipeline
    parallel size and pipeline parallel rank.
    Args:
        args (argparse.Namespace): the arguments to the script
        tp_size (int): the tensor parallel size
        pp_size (int): the pipeline parallel size
        pp_rank (int): the pipeline parallel rank
    """
    tp_state_dicts = []
    for i in range(tp_size):
        sub_dir_name = f"mp_rank_{i:02d}" if pp_size == 1 else f"mp_rank_{i:02d}_{pp_rank:03d}"
        checkpoint_name = os.listdir(os.path.join(args.load_path, sub_dir_name))[0]
        checkpoint_path = os.path.join(args.load_path, sub_dir_name, checkpoint_name)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        tp_state_dicts.append(state_dict)
    return tp_state_dicts


def megatron_to_transformers_fix_query_key_value_ordering(
        param, checkpoint_version, num_splits, num_heads, hidden_size
):
    """
    Permutes layout of param tensor to [num_splits * num_heads * hidden_size, :] for compatibility with later versions
    of NVIDIA Megatron-LM. The inverse operation is performed inside Megatron-LM to read checkpoints:
    https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209 If param is the weight tensor of the
    self-attention block, the returned tensor will have to be transposed one more time to be read by HuggingFace GPT2.
    This function is taken from `convert_megatron_gpt2_checkpoint.py`
    Args:
        param (torch.Tensor): the tensor to permute
        checkpoint_version (int): the version of the checkpoint.
        num_splits (int): the number of projections, usually 3 for (Query, Key, Value)
        num_heads (int): the number of attention heads
        hidden_size (int): the hidden size per head
    """

    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


def transformers_to_megatron_fix_query_key_value_ordering(
    param, checkpoint_version, num_splits, num_heads, hidden_size
):
    """
    Permutes layout of param tensor to the one compatible with respective NVIDIA Megatron-LM chekpoint versions. Input
    is [num_splits * num_heads * hidden_size, :] and output is [num_heads * hidden_size * num_splits, :] for version
    1.0 and [num_heads * num_splits * hidden_size, :] for version 2.0 and later. If param is the weight tensor of the
    self-attention block, the param needs to be already transposed before calling this function.
    Args:
        param (torch.Tensor): the tensor to permute
        checkpoint_version (int): the version of the checkpoint.
        num_splits (int): the number of projections, usually 3 for (Query, Key, Value)
        num_heads (int): the number of attention heads
        hidden_size (int): the hidden size per head
    """

    # Input is [num_splits * num_heads * hidden_size, :]
    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        current_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
        param = param.view(*current_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        current_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
        param = param.view(*current_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


def recursive_print(name, val, spaces=0):
    """
    Recursively print the structure of a checkpoint. This function is taken from `convert_megatron_gpt2_checkpoint.py`
    Args:
        name (str): the name of the current tensor parameter
        val (Tuple(int)): the shape of the current tensor parameter
        spaces (int): the number of spaces to print before the output for a nested structure
    """
    # Format the message.
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


def get_element_from_dict_by_path(d, path):
    if path not in d:
        d[path] = {}
    d = d[path]
    return d


def convert_checkpoint_from_transformers_to_megatron(args):
    assert args.world_size == args.target_expert_model_parallel_size * args.target_tensor_model_parallel_size * args.target_pipeline_model_parallel_size

    os.makedirs(args.save_path, exist_ok=True)

    # Saving config and tokenzier files
    os.system("cp -rf " + args.load_path + "/*.json " + args.save_path)
    os.system("cp -rf " + args.load_path + "/tokeniz* " + args.save_path)
    os.system("cp -rf " + args.load_path + "/merges.txt " + args.save_path)

    # Saving the tracker file
    tracker_filepath = os.path.join(args.save_path, "latest_checkpointed_iteration.txt")
    with open(tracker_filepath, "w") as f:
        f.write("release")

    # create `release` dir in args.load_path
    release_dir = os.path.join(args.save_path, "release")
    os.makedirs(release_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(args.load_path)

    if args.model_name == "qwen1.5-0.5B":
        state_dict = AutoModelForCausalLM.from_pretrained(args.load_path).state_dict()
    else:
        raise ValueError("model name is not supported")

    internal_state_dict = {}
    for layer_id in range(config.num_hidden_layers):

        q_weight = state_dict['model.layers.'+str(layer_id)+'.self_attn.q_proj.weight']
        k_weight = state_dict['model.layers.' + str(layer_id) + '.self_attn.k_proj.weight']
        v_weight = state_dict['model.layers.' + str(layer_id) + '.self_attn.v_proj.weight']

        q_bias = state_dict['model.layers.'+str(layer_id)+'.self_attn.q_proj.bias']
        k_bias = state_dict['model.layers.' + str(layer_id) + '.self_attn.k_proj.bias']
        v_bias = state_dict['model.layers.' + str(layer_id) + '.self_attn.v_proj.bias']

        internal_state_dict['transformer.layers.'+str(layer_id)+'.self_attn.query_key_value.weight'] =\
                torch.cat((q_weight, k_weight, v_weight))
        internal_state_dict['transformer.layers.'+str(layer_id)+'.self_attn.query_key_value.bias'] =\
                torch.cat((q_bias, k_bias, v_bias))

        internal_state_dict['transformer.layers.' + str(layer_id) + '.self_attn.dense.weight'] = \
            state_dict['model.layers.' + str(layer_id) + '.self_attn.o_proj.weight']

        dense_h_to_4h_1_weight = state_dict[
            'model.layers.' + str(layer_id) + '.mlp.gate_proj.weight']

        dense_h_to_4h_2_weight = state_dict[
            'model.layers.' + str(layer_id) + '.mlp.up_proj.weight']

        internal_state_dict['transformer.layers.' + str(layer_id) + '.mlp.dense_h_to_4h_1.weight'] =\
            dense_h_to_4h_1_weight

        internal_state_dict['transformer.layers.' + str(layer_id) + '.mlp.dense_h_to_4h_2.weight'] =\
            dense_h_to_4h_2_weight

        internal_state_dict['transformer.layers.' + str(layer_id) + '.mlp.dense_4h_to_h.weight'] = state_dict[
            'model.layers.' + str(layer_id) + '.mlp.down_proj.weight']

        internal_state_dict['transformer.layers.' + str(layer_id) + '.input_layernorm.weight'] = state_dict[
            'model.layers.' + str(layer_id) + '.input_layernorm.weight']

        internal_state_dict['transformer.layers.' + str(layer_id) + '.post_attention_layernorm.weight'] = state_dict[
            'model.layers.' + str(layer_id) + '.post_attention_layernorm.weight']

    internal_state_dict["transformer.word_embeddings.weight"] = state_dict['model.embed_tokens.weight']
    internal_state_dict["transformer.final_layernorm.weight"] = state_dict['model.norm.weight']
    internal_state_dict["transformer.lm_head.weight"] = state_dict['lm_head.weight']

    output_state_dict = []
    for i in range(args.target_tensor_model_parallel_size):
        output_state_dict.append(OrderedDict())


    if args.target_params_dtype == "fp16":
        dtype = torch.float16
    elif args.target_params_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # megatron args
    megatron_args = {
        "orig_vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "tensor_model_parallel_size": args.target_tensor_model_parallel_size,
        "pipeline_model_parallel_size": args.target_pipeline_model_parallel_size
    }

    margs = types.SimpleNamespace()
    for k, v in megatron_args.items():
        setattr(margs, k, v)

    # Embedding layer
    print("converting embedding layer")
    word_embedding = internal_state_dict["transformer.word_embeddings.weight"].to(dtype)
    out_word_embed = torch.chunk(word_embedding, args.target_tensor_model_parallel_size, dim=0)
    for i in range(args.target_tensor_model_parallel_size):
        word_emb_dict = get_element_from_dict_by_path(output_state_dict[i], "model")
        word_emb_dict["embedding.word_embeddings.weight"] = out_word_embed[i]

    print("converting output layer")
    lm_head = internal_state_dict["transformer.lm_head.weight"].to(dtype)
    out_lm_head = torch.chunk(lm_head, args.target_tensor_model_parallel_size, dim=0)
    for i in range(args.target_tensor_model_parallel_size):
        lm_head_dict = get_element_from_dict_by_path(output_state_dict[i], "model")
        lm_head_dict["output_layer.weight"] = out_lm_head[i]

    print("converting transformer layers")
    if config.num_hidden_layers % args.target_pipeline_model_parallel_size != 0:
        raise ValueError(
            f"Number of layers ({config.num_hidden_layers}) must be divisible by number of pipeline parallelism"
            f" ({args.target_pipeline_model_parallel_size})"
        )

    num_layers = config.num_hidden_layers // args.target_pipeline_model_parallel_size

    layer_re = re.compile("transformer.layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    num_heads = config.num_attention_heads
    hidden_size_per_head = config.hidden_size // config.num_attention_heads

    for pp_rank in range(args.target_pipeline_model_parallel_size):
        layer_offset = pp_rank * num_layers
        if pp_rank > 0:
            output_state_dict = []
            for i in range(args.target_tensor_model_parallel_size):
                output_state_dict.append({})

        for layer in range(num_layers):
            pp_layer_id = layer + layer_offset
            layers_to_copy = [
                layer_name
                for layer_name in internal_state_dict.keys()
                if layer_name.startswith(f"transformer.layers.{pp_layer_id}.")
            ]

            for layer_name in layers_to_copy:
                m = layer_re.match(layer_name)
                # Stop if that's not a layer
                if m is None:
                    break

                # The index of the layer.
                _ = int(m.group(1))
                # The name of the operation.
                op_name = m.group(2)
                # Is it a weight or a bias?
                weight_or_bias = m.group(3)

                params = internal_state_dict[layer_name].to(dtype)
                # handle layernorm
                if op_name.startswith("input_layernorm") and weight_or_bias == "weight":
                    out_name = "self_attention.linear_qkv.layer_norm_weight"
                    layer_name = f"layers.{layer}.{out_name}"

                elif op_name.startswith("post_attention_layernorm") and weight_or_bias == "weight":
                    out_name = "mlp.linear_fc1.layer_norm_weight"
                    layer_name = f"layers.{layer}.{out_name}"

                # handle attention K, V, Q weights and bias
                elif op_name.startswith("self_attn.query_key_value"):
                    # transformers stores D X (3*D) but Megatron-LM expects (3*D) X D.
                    params = transformers_to_megatron_fix_query_key_value_ordering(
                        params,
                        3.0,
                        3,
                        num_heads,
                        hidden_size_per_head,
                    )
                    layer_name = f"layers.{layer}.self_attention.linear_qkv.{weight_or_bias}"

                # handle attention and mlp weights
                elif weight_or_bias == "weight":
                    out_name = internal_to_output_mapping.get(op_name, None)
                    if out_name is None:
                        continue
                    layer_name = f"layers.{layer}.{out_name}.{weight_or_bias}"

                # skip
                else:
                    continue

                if op_name + "." + weight_or_bias in tensor_parallel_params:
                    dim = 1 if op_name + "." + weight_or_bias in column_split_tensor_parallel_params else 0
                    params = torch.chunk(params, args.target_tensor_model_parallel_size, dim=dim)

                for i in range(args.target_tensor_model_parallel_size):
                    params_dict = get_element_from_dict_by_path(output_state_dict[i], "model")
                    params_dict["decoder." + layer_name] = (
                        params[i].clone() if (
                                    op_name + "." + weight_or_bias in tensor_parallel_params) else params.clone()
                    )

            for i in range(args.target_tensor_model_parallel_size):

                params_dict = get_element_from_dict_by_path(output_state_dict[i], "model")
                dense_h_to_4h_1_name = f'decoder.layers.{layer}.mlp.linear_fc1_1.weight'
                dense_h_to_4h_1_weight = params_dict[dense_h_to_4h_1_name]

                dense_h_to_4h_2_name = f'decoder.layers.{layer}.mlp.linear_fc1_2.weight'
                dense_h_to_4h_2_weight = params_dict[dense_h_to_4h_2_name]

                dense_h_to_4h_name = f'decoder.layers.{layer}.mlp.linear_fc1.weight'
                params_dict[dense_h_to_4h_name] = torch.cat(
                [dense_h_to_4h_1_weight, dense_h_to_4h_2_weight], dim=0)

                del params_dict[dense_h_to_4h_1_name]
                del params_dict[dense_h_to_4h_2_name]

        if pp_rank == args.target_pipeline_model_parallel_size - 1:
            # handle final layernorm
            for weight_or_bias in ["weight"]:
                params = internal_state_dict[f"transformer.final_layernorm.{weight_or_bias}"].to(dtype)
                layer_name = "decoder." + f"final_layernorm.{weight_or_bias}"
                for i in range(args.target_tensor_model_parallel_size):
                    params_dict = get_element_from_dict_by_path(output_state_dict[i], "model")
                    params_dict[layer_name] = params.clone()

            # add the embedding
            for i in range(args.target_tensor_model_parallel_size):
                params_dict = get_element_from_dict_by_path(output_state_dict[i], "model")
                params_dict["embedding.word_embeddings.weight"] = out_word_embed[i].clone()

            # add the LM head
            for i in range(args.target_tensor_model_parallel_size):
                params_dict = get_element_from_dict_by_path(output_state_dict[i], "model")
                params_dict["output_layer.weight"] = out_lm_head[i].clone()

        # saving the state dict as per the tp_rank and pp_rank
        for tp_rank in range(args.target_tensor_model_parallel_size):
            output_state_dict[tp_rank]["checkpoint_version"] = 3.0
            output_state_dict[tp_rank]["args"] = margs
            checkpoint_dir = (
                f"mp_rank_{tp_rank:02d}"
                if args.target_pipeline_model_parallel_size == 1
                else f"mp_rank_{tp_rank:02d}_{pp_rank:03d}"
            )

            checkpoint_name = "model_optim_rng.pt"
            checkpoint_dir = os.path.join(release_dir, checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            if args.print_checkpoint_structure:
                print(
                    f"Checkpoint structure of model state dict shard belonging to TP rank {tp_rank} and PP rank"
                    f" {pp_rank}:"
                )
                recursive_print(None, output_state_dict[tp_rank])
            torch.save(output_state_dict[tp_rank], checkpoint_path)



def main():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    convert_checkpoint_from_transformers_to_megatron(args)


if __name__ == "__main__":
    main()
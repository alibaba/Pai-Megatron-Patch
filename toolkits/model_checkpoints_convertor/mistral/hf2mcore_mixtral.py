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
import json
import types
from collections import OrderedDict

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, MixtralConfig
from toolkits.model_checkpoints_convertor.utils import (
    save_state_dict,
    save_hfmodel
)

def add_args(parser):
    parser.add_argument('--megatron-path',
                        type=str,
                        default=None,
                        help='Base directory of Megatron repository')

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
    "mlp.megatron_moe.gate.wg": "mlp.router",
    "mlp.megatron_moe.experts.megatron_experts.0.dense_h_to_4h_1": "mlp.experts.local_experts.0.linear_fc1_1",
    "mlp.megatron_moe.experts.megatron_experts.1.dense_h_to_4h_1": "mlp.experts.local_experts.1.linear_fc1_1",
    "mlp.megatron_moe.experts.megatron_experts.2.dense_h_to_4h_1": "mlp.experts.local_experts.2.linear_fc1_1",
    "mlp.megatron_moe.experts.megatron_experts.3.dense_h_to_4h_1": "mlp.experts.local_experts.3.linear_fc1_1",
    "mlp.megatron_moe.experts.megatron_experts.4.dense_h_to_4h_1": "mlp.experts.local_experts.4.linear_fc1_1",
    "mlp.megatron_moe.experts.megatron_experts.5.dense_h_to_4h_1": "mlp.experts.local_experts.5.linear_fc1_1",
    "mlp.megatron_moe.experts.megatron_experts.6.dense_h_to_4h_1": "mlp.experts.local_experts.6.linear_fc1_1",
    "mlp.megatron_moe.experts.megatron_experts.7.dense_h_to_4h_1": "mlp.experts.local_experts.7.linear_fc1_1",
    "mlp.megatron_moe.experts.megatron_experts.0.dense_h_to_4h_2": "mlp.experts.local_experts.0.linear_fc1_2",
    "mlp.megatron_moe.experts.megatron_experts.1.dense_h_to_4h_2": "mlp.experts.local_experts.1.linear_fc1_2",
    "mlp.megatron_moe.experts.megatron_experts.2.dense_h_to_4h_2": "mlp.experts.local_experts.2.linear_fc1_2",
    "mlp.megatron_moe.experts.megatron_experts.3.dense_h_to_4h_2": "mlp.experts.local_experts.3.linear_fc1_2",
    "mlp.megatron_moe.experts.megatron_experts.4.dense_h_to_4h_2": "mlp.experts.local_experts.4.linear_fc1_2",
    "mlp.megatron_moe.experts.megatron_experts.5.dense_h_to_4h_2": "mlp.experts.local_experts.5.linear_fc1_2",
    "mlp.megatron_moe.experts.megatron_experts.6.dense_h_to_4h_2": "mlp.experts.local_experts.6.linear_fc1_2",
    "mlp.megatron_moe.experts.megatron_experts.7.dense_h_to_4h_2": "mlp.experts.local_experts.7.linear_fc1_2",
    "mlp.megatron_moe.experts.megatron_experts.0.dense_h_to_4h": "mlp.experts.local_experts.0.linear_fc1",
    "mlp.megatron_moe.experts.megatron_experts.1.dense_h_to_4h": "mlp.experts.local_experts.1.linear_fc1",
    "mlp.megatron_moe.experts.megatron_experts.2.dense_h_to_4h": "mlp.experts.local_experts.2.linear_fc1",
    "mlp.megatron_moe.experts.megatron_experts.3.dense_h_to_4h": "mlp.experts.local_experts.3.linear_fc1",
    "mlp.megatron_moe.experts.megatron_experts.4.dense_h_to_4h": "mlp.experts.local_experts.4.linear_fc1",
    "mlp.megatron_moe.experts.megatron_experts.5.dense_h_to_4h": "mlp.experts.local_experts.5.linear_fc1",
    "mlp.megatron_moe.experts.megatron_experts.6.dense_h_to_4h": "mlp.experts.local_experts.6.linear_fc1",
    "mlp.megatron_moe.experts.megatron_experts.7.dense_h_to_4h": "mlp.experts.local_experts.7.linear_fc1",
    "mlp.megatron_moe.experts.megatron_experts.0.dense_4h_to_h": "mlp.experts.local_experts.0.linear_fc2",
    "mlp.megatron_moe.experts.megatron_experts.1.dense_4h_to_h": "mlp.experts.local_experts.1.linear_fc2",
    "mlp.megatron_moe.experts.megatron_experts.2.dense_4h_to_h": "mlp.experts.local_experts.2.linear_fc2",
    "mlp.megatron_moe.experts.megatron_experts.3.dense_4h_to_h": "mlp.experts.local_experts.3.linear_fc2",
    "mlp.megatron_moe.experts.megatron_experts.4.dense_4h_to_h": "mlp.experts.local_experts.4.linear_fc2",
    "mlp.megatron_moe.experts.megatron_experts.5.dense_4h_to_h": "mlp.experts.local_experts.5.linear_fc2",
    "mlp.megatron_moe.experts.megatron_experts.6.dense_4h_to_h": "mlp.experts.local_experts.6.linear_fc2",
    "mlp.megatron_moe.experts.megatron_experts.7.dense_4h_to_h": "mlp.experts.local_experts.7.linear_fc2",
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
    "mlp.megatron_moe.experts.megatron_experts.0.dense_h_to_4h_1.weight",
    "mlp.megatron_moe.experts.megatron_experts.1.dense_h_to_4h_1.weight",
    "mlp.megatron_moe.experts.megatron_experts.2.dense_h_to_4h_1.weight",
    "mlp.megatron_moe.experts.megatron_experts.3.dense_h_to_4h_1.weight",
    "mlp.megatron_moe.experts.megatron_experts.4.dense_h_to_4h_1.weight",
    "mlp.megatron_moe.experts.megatron_experts.5.dense_h_to_4h_1.weight",
    "mlp.megatron_moe.experts.megatron_experts.6.dense_h_to_4h_1.weight",
    "mlp.megatron_moe.experts.megatron_experts.7.dense_h_to_4h_1.weight",
    "mlp.megatron_moe.experts.megatron_experts.0.dense_h_to_4h_2.weight",
    "mlp.megatron_moe.experts.megatron_experts.1.dense_h_to_4h_2.weight",
    "mlp.megatron_moe.experts.megatron_experts.2.dense_h_to_4h_2.weight",
    "mlp.megatron_moe.experts.megatron_experts.3.dense_h_to_4h_2.weight",
    "mlp.megatron_moe.experts.megatron_experts.4.dense_h_to_4h_2.weight",
    "mlp.megatron_moe.experts.megatron_experts.5.dense_h_to_4h_2.weight",
    "mlp.megatron_moe.experts.megatron_experts.6.dense_h_to_4h_2.weight",
    "mlp.megatron_moe.experts.megatron_experts.7.dense_h_to_4h_2.weight",
    "mlp.megatron_moe.experts.megatron_experts.0.dense_4h_to_h.weight",
    "mlp.megatron_moe.experts.megatron_experts.1.dense_4h_to_h.weight",
    "mlp.megatron_moe.experts.megatron_experts.2.dense_4h_to_h.weight",
    "mlp.megatron_moe.experts.megatron_experts.3.dense_4h_to_h.weight",
    "mlp.megatron_moe.experts.megatron_experts.4.dense_4h_to_h.weight",
    "mlp.megatron_moe.experts.megatron_experts.5.dense_4h_to_h.weight",
    "mlp.megatron_moe.experts.megatron_experts.6.dense_4h_to_h.weight",
    "mlp.megatron_moe.experts.megatron_experts.7.dense_4h_to_h.weight"
]

tensor_parallel_params_mg = [
    # megatron-lm layers to merge across tp ranks
    'self_attention.linear_proj.weight',
    'self_attention.linear_qkv.weight',
]

column_split_tensor_parallel_params = [
    "self_attn.dense.weight",
    "mlp.megatron_moe.experts.megatron_experts.0.dense_4h_to_h.weight",
    "mlp.megatron_moe.experts.megatron_experts.1.dense_4h_to_h.weight",
    "mlp.megatron_moe.experts.megatron_experts.2.dense_4h_to_h.weight",
    "mlp.megatron_moe.experts.megatron_experts.3.dense_4h_to_h.weight",
    "mlp.megatron_moe.experts.megatron_experts.4.dense_4h_to_h.weight",
    "mlp.megatron_moe.experts.megatron_experts.5.dense_4h_to_h.weight",
    "mlp.megatron_moe.experts.megatron_experts.6.dense_4h_to_h.weight",
    "mlp.megatron_moe.experts.megatron_experts.7.dense_4h_to_h.weight"
]

column_split_tensor_parallel_params_mg = [
    'self_attention.linear_proj'
]


def get_checkpoint_sub_dir_name(tp_rank, pp_rank, pp_size, ep_rank, ep_size):
    sub_dir_name = f"mp_rank_{tp_rank:02d}"
    if pp_size > 1: sub_dir_name = f"{sub_dir_name}_{pp_rank:03d}"
    if ep_size > 1: sub_dir_name = f"{sub_dir_name}_{ep_rank:03d}"
    return sub_dir_name


def get_megatron_sharded_states(args, tp_size, pp_size, ep_size, pp_rank):
    """
    Get sharded checkpoints from NVIDIA Megatron-LM checkpoint based on the provided tensor parallel size, pipeline
    parallel size and pipeline parallel rank.
    Args:
        args (argparse.Namespace): the arguments to the script
        tp_size (int): the tensor parallel size
        pp_size (int): the pipeline parallel size
        pp_rank (int): the pipeline parallel rank
    """
    tp_state_dicts = [{'model': {}} for i in range(tp_size)]
    global_ep_index = 0
    for tp_index, i in enumerate(range(tp_size)):
        for ep_index, j in enumerate(range(ep_size)):
            sub_dir_name = get_checkpoint_sub_dir_name(i, pp_rank, pp_size, j, ep_size)
            print(f"Loading {sub_dir_name}...")
            checkpoint_name = os.listdir(os.path.join(args.load_path, sub_dir_name))[0]
            checkpoint_path = os.path.join(args.load_path, sub_dir_name, checkpoint_name)
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            ep_length = len([i for i in state_dict['model'] if 'linear_fc2.weight' in i and 'decoder.layers.0' in i])
            # combine experts within a tensor_parallel
            for key, value in list(state_dict['model'].items()):
                if 'linear_fc' in key:
                    key_list = key.split('.')
                    local_ep_index = int(key_list[6])
                    key_list[6] = str(ep_index * ep_length + local_ep_index)
                    del state_dict['model'][key]
                    state_dict['model']['.'.join(key_list)] = value
            tp_state_dicts[tp_index]['model'].update(state_dict['model'])
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

    # Saving the tracker file
    tracker_filepath = os.path.join(args.save_path, "latest_checkpointed_iteration.txt")
    with open(tracker_filepath, "w") as f:
        f.write("release")

    # create `release` dir in args.load_path
    release_dir = os.path.join(args.save_path, "release")
    os.makedirs(release_dir, exist_ok=True)
    config = AutoConfig.from_pretrained(args.load_path)
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

    state_dict = AutoModelForCausalLM.from_pretrained(args.load_path).state_dict()
    internal_state_dict = {}
    for layer_id in range(config.num_hidden_layers):

        q_weight = state_dict['model.layers.' + str(layer_id) + '.self_attn.q_proj.weight']
        k_weight = state_dict['model.layers.' + str(layer_id) + '.self_attn.k_proj.weight']
        v_weight = state_dict['model.layers.' + str(layer_id) + '.self_attn.v_proj.weight']

        internal_state_dict['transformer.layers.' + str(layer_id) + '.self_attn.query.weight'] = q_weight
        internal_state_dict['transformer.layers.' + str(layer_id) + '.self_attn.key_value.weight'] = torch.cat(
            (k_weight, v_weight))

        internal_state_dict['transformer.layers.' + str(layer_id) + '.self_attn.dense.weight'] = \
            state_dict['model.layers.' + str(layer_id) + '.self_attn.o_proj.weight']

        internal_state_dict['transformer.layers.' + str(layer_id) + '.mlp.megatron_moe.gate.wg.weight'] = state_dict[
            'model.layers.' + str(layer_id) + '.block_sparse_moe.gate.weight']

        for expert_id in range(config.num_local_experts):
            internal_state_dict[
                'transformer.layers.' + str(layer_id) + '.mlp.megatron_moe.experts.megatron_experts.' + str(
                    expert_id) + '.dense_h_to_4h_1.weight'] = \
                state_dict[
                    'model.layers.' + str(layer_id) + '.block_sparse_moe.experts.' + str(expert_id) + '.w1.weight']

            internal_state_dict[
                'transformer.layers.' + str(layer_id) + '.mlp.megatron_moe.experts.megatron_experts.' + str(
                    expert_id) + '.dense_h_to_4h_2.weight'] = \
                state_dict[
                    'model.layers.' + str(layer_id) + '.block_sparse_moe.experts.' + str(expert_id) + '.w3.weight']

            internal_state_dict[
                'transformer.layers.' + str(layer_id) + '.mlp.megatron_moe.experts.megatron_experts.' + str(
                    expert_id) + '.dense_4h_to_h.weight'] = state_dict[
                'model.layers.' + str(layer_id) + '.block_sparse_moe.experts.' + str(expert_id) + '.w2.weight']

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

    num_query_group = config.num_key_value_heads
    output_group_state_dict = []
    for i in range(num_query_group):
        output_group_state_dict.append({})

    if args.target_params_dtype == "fp16":
        dtype = torch.float16
    elif args.target_params_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

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

    hidden_size = config.hidden_size
    num_groups = config.num_key_value_heads
    num_heads = config.num_attention_heads
    hidden_size_per_head = config.hidden_size // config.num_attention_heads

    for pp_rank in range(args.target_pipeline_model_parallel_size):
        layer_offset = pp_rank * num_layers
        if pp_rank > 0:
            output_state_dict = []
            for i in range(args.target_tensor_model_parallel_size):
                output_state_dict.append({})

            output_group_state_dict = []
            for i in range(num_query_group):
                output_group_state_dict.append({})

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
                extra_state_name = None
                if op_name.startswith("input_layernorm") and weight_or_bias == "weight":
                    out_name = "self_attention.linear_qkv"
                    layer_name = f"layers.{layer}.{out_name}.layer_norm_weight"
                    extra_state_name = f"layers.{layer}.{out_name}._extra_state"
                elif op_name.startswith("post_attention_layernorm") and weight_or_bias == "weight":
                    out_name = "pre_mlp_layernorm"
                    layer_name = f"layers.{layer}.{out_name}.weight"
                    extra_state_name = f"layers.{layer}.{out_name}._extra_state"
                # handle attention K, V, Q weights
                elif op_name.startswith("self_attn.query") and weight_or_bias == "weight":
                    # transformers stores D X (3*D) but Megatron-LM expects (3*D) X D.
                    params = transformers_to_megatron_fix_query_key_value_ordering(
                        params,
                        3.0,
                        1,
                        num_heads,
                        hidden_size_per_head,
                    )
                    layer_name = f"layers.{layer}.{op_name}.{weight_or_bias}"
                elif op_name.startswith("self_attn.key_value") and weight_or_bias == "weight":
                    # transformers stores D X (3*D) but Megatron-LM expects (3*D) X D.
                    params = transformers_to_megatron_fix_query_key_value_ordering(
                        params,
                        3.0,
                        2,
                        num_groups,
                        hidden_size_per_head,
                    )
                    layer_name = f"layers.{layer}.{op_name}.{weight_or_bias}"
                # handle attention and mlp weights
                elif weight_or_bias == "weight":
                    out_name = internal_to_output_mapping.get(op_name, None)
                    if out_name is None:
                        continue
                    layer_name = f"layers.{layer}.{out_name}.{weight_or_bias}"
                    if out_name not in ['linear_fc1_1', 'mlp.router', 'linear_fc1_2']:
                        extra_state_name = f"layers.{layer}.{out_name}._extra_state"
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
                    if extra_state_name is not None:
                        if 'linear_fc1_' not in extra_state_name:
                            params_dict["decoder." + extra_state_name] = None

            for i in range(args.target_tensor_model_parallel_size):

                params_dict = get_element_from_dict_by_path(output_state_dict[i], "model")
                for expert_id in range(config.num_local_experts):
                    dense_h_to_4h_1_name = f'decoder.layers.{layer}.mlp.experts.local_experts.{expert_id}.linear_fc1_1.weight'
                    dense_h_to_4h_1_weight = params_dict[dense_h_to_4h_1_name]
                    del params_dict[dense_h_to_4h_1_name]

                    dense_h_to_4h_2_name = f'decoder.layers.{layer}.mlp.experts.local_experts.{expert_id}.linear_fc1_2.weight'
                    dense_h_to_4h_2_weight = params_dict[dense_h_to_4h_2_name]
                    del params_dict[dense_h_to_4h_2_name]

                    dense_h_to_4h_name = f'decoder.layers.{layer}.mlp.experts.local_experts.{expert_id}.linear_fc1.weight'
                    params_dict[dense_h_to_4h_name] = \
                        torch.cat([dense_h_to_4h_1_weight, dense_h_to_4h_2_weight], dim=0)

                self_attn_query_name = f"decoder.layers.{layer}.self_attn.query.weight"
                query_weight = params_dict[self_attn_query_name]
                del params_dict[self_attn_query_name]
                self_attn_kv_name = f"decoder.layers.{layer}.self_attn.key_value.weight"
                kv_weight = params_dict[self_attn_kv_name]
                del params_dict[self_attn_kv_name]

                # torch.Size([8 512, 4096])
                group_query_weight = query_weight.view(num_groups // args.target_tensor_model_parallel_size,
                                                       num_heads // num_groups * hidden_size_per_head, hidden_size)
                # torch.Size(8, 256, 4096])
                group_kv_weight = kv_weight.view(num_groups // args.target_tensor_model_parallel_size,
                                                 2 * hidden_size_per_head, hidden_size)
                group_qkv_weight = torch.cat([group_query_weight, group_kv_weight], dim=1)
                params_dict["decoder." + f"layers.{layer}.self_attention.linear_qkv.weight"] = \
                    group_qkv_weight.view(-1, hidden_size)
                params_dict["decoder." + f"layers.{layer}.self_attention.linear_qkv._extra_state"] = None

        if pp_rank == args.target_pipeline_model_parallel_size - 1:
            # handle final layernorm
            for weight_or_bias in ["weight"]:
                params = internal_state_dict[f"transformer.final_layernorm.{weight_or_bias}"].to(dtype)
                layer_name = "decoder." + f"final_layernorm.{weight_or_bias}"
                for i in range(args.target_tensor_model_parallel_size):
                    params_dict = get_element_from_dict_by_path(output_state_dict[i], "model")
                    params_dict[layer_name] = params.clone()
                    params_dict["decoder.final_layernorm._extra_state"] = None

            # add the embedding
            for i in range(args.target_tensor_model_parallel_size):
                params_dict = get_element_from_dict_by_path(output_state_dict[i], "model")
                params_dict["embedding.word_embeddings.weight"] = out_word_embed[i].clone()

            # add the LM head
            for i in range(args.target_tensor_model_parallel_size):
                params_dict = get_element_from_dict_by_path(output_state_dict[i], "model")
                params_dict["output_layer.weight"] = out_lm_head[i].clone()

        num_ep_groups = args.world_size // args.target_tensor_model_parallel_size // args.target_pipeline_model_parallel_size
        experts_ids = [x for x in range(config.num_local_experts)]
        chunks = [experts_ids[x:x + config.num_local_experts // num_ep_groups] for x in
                  range(0, len(experts_ids), config.num_local_experts // num_ep_groups)]

        expert_group_mapping = {}
        for idx, chunk in enumerate(chunks):
            for ele in chunk:
                expert_group_mapping[ele] = idx

        expert_local_mapping = {}
        for chunk in chunks:
            for idx, ele in enumerate(chunk):
                expert_local_mapping[ele] = idx

        # saving the state dict as per the tp_rank and pp_rank
        for tp_rank in range(args.target_tensor_model_parallel_size):
            current_keys = list(output_state_dict[tp_rank]['model'].keys())
            ep_state_dict = []
            for i in range(args.target_expert_model_parallel_size):
                ep_state_dict.append({})
            for key in current_keys:
                if "local_experts" in key:
                    keywords = key.split(".")
                    eid = int(keywords[6])
                    expert_group_id = expert_group_mapping[eid]
                    local_expert_id = expert_local_mapping[eid]
                    keywords[6] = str(local_expert_id)

                    ep_state_dict[expert_group_id][".".join(keywords)] = (
                        output_state_dict[tp_rank]['model'][key].clone() 
                        if hasattr(output_state_dict[tp_rank]['model'][key], 'clone') 
                        else output_state_dict[tp_rank]['model'][key]
                    )
                    output_state_dict[tp_rank]['model'].pop(key)

            for ep_rank in range(args.target_expert_model_parallel_size):
                checkpoint_dir = get_checkpoint_sub_dir_name(tp_rank, pp_rank, args.target_pipeline_model_parallel_size,
                                                             ep_rank, args.target_expert_model_parallel_size)
                save_dir = os.path.join(release_dir, checkpoint_dir)
                os.makedirs(save_dir, exist_ok=True)
                checkpoint_name = "model_optim_rng.pt"
                checkpoint_path = os.path.join(save_dir, checkpoint_name)
                output_state_dict[tp_rank]['model'].update(ep_state_dict[ep_rank])
                save_state_dict(args, [output_state_dict[tp_rank]['model']], checkpoint_path, save_args=False)


def convert_checkpoint_from_megatron_to_transformers(args):
    """
    Convert NVIDIA Megatron-LM checkpoint to HuggingFace Transformers checkpoint. This handles Megatron checkpoints
    with different tensor parallelism and pipeline parallelism sizes. It saves the converted checkpoint into shards
    using HuggingFace Transformers checkpoint sharding functionality. This greatly extends the functionality of
    `convert_megatron_gpt2_checkpoint.py`

    Args:
        args (argparse.Namespace): the arguments to the script
    """
    os.makedirs(args.save_path, exist_ok=True)

    # Saving config and tokenzier files
    os.system("cp -rf " + args.load_path + "/*.json " + args.save_path)
    os.system("cp -rf " + args.load_path + "/tokenizer.model " + args.save_path)

    tracker_filepath = os.path.join(args.load_path, "latest_checkpointed_iteration.txt")
    with open(tracker_filepath, "r") as f:
        tag = f.readline()

    args.load_path = os.path.join(args.load_path, tag)
    import glob
    if glob.glob(args.load_path + "/mp_rank*/distrib*"):
        # if os.path.exists(args.load_path+"/mp_rank*/distrib*"):
        user_input = input(
            "Optimizer states detected. Will remove distrib* files. yes (remove and continue) / no (stop programme): ")
        if user_input == 'yes':
            os.system("rm -rf " + args.load_path + "/mp_rank*/distrib*")
        else:
            raise RuntimeError("Optimizer states are not removed. Save files to another folder and re-run.")

    # params dtype
    if args.target_params_dtype == "fp16":
        dtype = torch.float16
    elif args.target_params_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    config = MixtralConfig()

    output_state_dict = {}

    # checkpoint_version = state_dict.get("checkpoint_version", 3.0)
    tp_size = args.target_tensor_model_parallel_size
    pp_size = args.target_pipeline_model_parallel_size
    ep_size = args.target_expert_model_parallel_size

    # The regex to extract layer names.
    layer_re = re.compile("layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # Convert.
    print("Converting")

    # Embeddings
    print("Converting embeddings")
    tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, ep_size, 0)

    # import pdb
    # pdb.set_trace()

    # Convert and store the word embeddings.
    word_embeddings = []
    word_embeddings_layernorm_weight = []
    word_embeddings_layernorm_bias = []

    # import pdb
    # pdb.set_trace()
    embeddings = tp_state_dicts[0]["model"]["embedding.word_embeddings.weight"]
    for tp_rank in range(tp_size):
        embeddings = tp_state_dicts[tp_rank]["model"]["embedding.word_embeddings.weight"]
        word_embeddings.append(embeddings)

    word_embeddings = torch.cat(word_embeddings, dim=0)
    word_embeddings = word_embeddings.to(dtype)
    output_state_dict["model.embed_tokens.weight"] = word_embeddings.clone()
    # Reset the vocab size
    config.vocab_size = word_embeddings.shape[0]

    # Transformer Layers
    print("Converting transformer layers")
    # The number of heads.
    heads = config.num_attention_heads
    # The hidden_size per head.
    hidden_size_per_head = config.hidden_size // config.num_attention_heads
    num_layers = config.num_hidden_layers // pp_size

    hidden_size = config.hidden_size
    num_groups = config.num_key_value_heads

    for pp_rank in range(pp_size):
        if pp_size > 0:
            print(f"Converting pipeline parallel rank {pp_rank}")
            tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, ep_size, pp_rank)

        # The transformer.

        path = 'model'

        # Extract the layers.
        for key, val in get_element_from_dict_by_path(tp_state_dicts[0], path).items():
            if key.endswith('_extra_state'):
                continue
            # deal with experts
            if 'linear_fc' in key:
                print(key)
                key_list = key.split('.')
                layer_id = int(key_list[2]) + pp_rank * num_layers
                expert_id = key_list[-3]
                dim = 1 if 'linear_fc2' in key else 0
                params = torch.cat(
                    [val]
                    + [
                        get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key]
                        for tp_rank in range(1, tp_size)
                    ],
                    dim=dim,
                ).to(dtype)

                if 'linear_fc2' in key:
                    output_state_dict[
                        f'model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}.w2.weight'] = params
                else:
                    params_split = [torch.chunk(i, 2, 0) for i in torch.chunk(params, tp_size, 0)]
                    output_state_dict[
                        f'model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}.w1.weight'] = torch.cat(
                        [i[0] for i in params_split])
                    output_state_dict[
                        f'model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}.w3.weight'] = torch.cat(
                        [i[1] for i in params_split])

                continue

            new_key = key.replace('decoder.', '')
            if 'layer_norm_weight' in new_key:
                new_key += '.weight'
            # Match the name.
            m = layer_re.match(new_key)
            # Stop if that's not a layer
            if m is None:
                continue

            # The index of the layer.
            layer_idx = int(m.group(1)) + pp_rank * num_layers
            # The name of the operation.
            op_name = m.group(2)
            # Is it a weight or a bias?
            weight_or_bias = m.group(3)

            # The name of the layer.
            layer_name = f"model.layers.{layer_idx}"

            print(layer_name, op_name, weight_or_bias)

            if op_name + "." + weight_or_bias not in tensor_parallel_params_mg:
                params = val.to(dtype)
            else:
                dim = 1 if op_name in column_split_tensor_parallel_params_mg else 0
                params = torch.cat(
                    [val]
                    + [
                        get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key]
                        for tp_rank in range(1, tp_size)
                    ],
                    dim=dim,
                ).to(dtype)

            # For layernorm(s), simply store the layer norm.
            if op_name.endswith("layer_norm_weight") or op_name.endswith("layernorm"):
                ln_name = "input_layernorm" if op_name.endswith("layer_norm_weight") else "post_attention_layernorm"
                output_state_dict[layer_name + "." + ln_name + "." + weight_or_bias] = params.clone()
                continue

            # Transpose the QKV matrix.
            elif (
                    op_name == "attention.linear_qkv" or op_name == "self_attention.linear_qkv"
            ) and weight_or_bias == "weight":

                all_qkvs = [i.reshape(num_groups // args.target_tensor_model_parallel_size,
                                      (heads // num_groups * hidden_size_per_head + 2 * hidden_size_per_head),
                                      hidden_size) for i in
                            torch.chunk(params, args.target_tensor_model_parallel_size, 0)]
                split_size = heads // num_groups * hidden_size_per_head
                all_qs = torch.cat([i[:, :split_size, :].reshape(-1, hidden_size) for i in all_qkvs])
                all_kvs = torch.cat([i[:, split_size:, :].reshape(-1, hidden_size) for i in all_qkvs])

                checkpoint_version = 3.0
                out_q = megatron_to_transformers_fix_query_key_value_ordering(
                    all_qs,
                    checkpoint_version,
                    1,
                    heads,
                    hidden_size_per_head,
                )

                out_kv = megatron_to_transformers_fix_query_key_value_ordering(
                    all_kvs,
                    checkpoint_version,
                    2,
                    num_groups,
                    hidden_size_per_head,
                )
                out_kv = torch.chunk(out_kv, 2)

                output_state_dict[layer_name + f".self_attn.q_proj.weight"] = out_q.clone()
                output_state_dict[layer_name + f".self_attn.k_proj.weight"] = out_kv[0].clone()
                output_state_dict[layer_name + f".self_attn.v_proj.weight"] = out_kv[1].clone()

            # Transpose the weights.
            elif weight_or_bias == "weight":
                out_name = megatron_to_transformers[op_name]
                output_state_dict[layer_name + '.' + out_name + '.' + "weight"] = params.clone()

    if config.num_hidden_layers != (layer_idx + 1):
        raise ValueError(f"Expected {config.num_hidden_layers} layers but found {layer_idx + 1}")

    # The final layernorm.
    print("Converting final layernorm")
    params = get_element_from_dict_by_path(tp_state_dicts[0], str(path))
    try:
        output_state_dict["model.norm.weight"] = params["decoder.final_layernorm.weight"].to(dtype).clone()
    except:
        output_state_dict["model.norm.weight"] = params["decoder.final_norm.weight"].to(dtype).clone()

    # For LM head, transformers' wants the matrix to weight embeddings.
    print("Converting LM head")
    params = torch.cat([
        get_element_from_dict_by_path(tp_state_dicts[i]['model'], 'output_layer.weight')
        for i in range(tp_size)]
    )
    output_state_dict["lm_head.weight"] = params.to(dtype).clone()

    # It should be done!
    print("Conversion from Megatron-LM to Transformers is done!")

    # Print the structure of converted state dict.
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)

    max_shard_size = int(args.max_shard_size) if args.max_shard_size.isdigit() else args.max_shard_size
    args.save_safetensors = False
    save_hfmodel(args, output_state_dict, max_shard_size)

def main():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    args.save = args.save_path
    if args.convert_checkpoint_from_megatron_to_transformers:
        convert_checkpoint_from_megatron_to_transformers(args)
    else:
        convert_checkpoint_from_transformers_to_megatron(args)


if __name__ == "__main__":
    main()
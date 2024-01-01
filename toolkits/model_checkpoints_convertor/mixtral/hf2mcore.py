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
import types
import torch
from collections import OrderedDict

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

def add_args(parser):

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
        "--target_params_dtype",
        type=str,
        default="fp32",
        help=(
            "The dtype of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )

    return parser

internal_to_output_mapping = {
    "self_attn.dense": "self_attention.linear_proj",
    "mlp.megatron_moe.gate.wg": "mlp.router.gate",
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

def get_element_from_dict_by_path(d, path):
    if path not in d:
        d[path] = {}
    d = d[path]
    return d

def convert_checkpoint_from_transformers_to_megatron(args):

    os.makedirs(args.save_path, exist_ok=True)

    # Saving config and tokenzier files
    os.system("cp -rf "+args.load_path+"/*.json "+args.save_path)
    os.system("cp -rf " + args.load_path + "/tokenizer* " + args.save_path)

    # Saving the tracker file
    tracker_filepath = os.path.join(args.save_path, "latest_checkpointed_iteration.txt")
    with open(tracker_filepath, "w") as f:
        f.write("release")

    # create `release` dir in args.load_path
    release_dir = os.path.join(args.save_path, "release")
    os.makedirs(release_dir, exist_ok=True)


    if args.model_name == "mixtral-8x7b":
        state_dict = AutoModelForCausalLM.from_pretrained(args.load_path).state_dict()
    else:
        raise ValueError("model name is not supported")

    config = AutoConfig.from_pretrained(args.load_path)

    internal_state_dict = {}
    for layer_id in range(config.num_hidden_layers):

        q_weight = state_dict['model.layers.'+str(layer_id)+'.self_attn.q_proj.weight']
        k_weight = state_dict['model.layers.' + str(layer_id) + '.self_attn.k_proj.weight']
        v_weight = state_dict['model.layers.' + str(layer_id) + '.self_attn.v_proj.weight']

        internal_state_dict['transformer.layers.'+str(layer_id)+'.self_attn.query.weight'] = q_weight
        internal_state_dict['transformer.layers.'+str(layer_id)+'.self_attn.key_value.weight'] = torch.cat((k_weight, v_weight))

        internal_state_dict['transformer.layers.' + str(layer_id) + '.self_attn.dense.weight'] =\
            state_dict['model.layers.' + str(layer_id) + '.self_attn.o_proj.weight']

        internal_state_dict['transformer.layers.' + str(layer_id) + '.mlp.megatron_moe.gate.wg.weight'] = state_dict[
            'model.layers.' + str(layer_id) + '.block_sparse_moe.gate.weight']

        for expert_id in range(config.num_local_experts):

            internal_state_dict['transformer.layers.' + str(layer_id) + '.mlp.megatron_moe.experts.megatron_experts.' + str(expert_id)+'.dense_h_to_4h_1.weight'] = \
                state_dict['model.layers.' + str(layer_id) + '.block_sparse_moe.experts.' + str(expert_id) + '.w1.weight']

            internal_state_dict['transformer.layers.' + str(layer_id) + '.mlp.megatron_moe.experts.megatron_experts.' + str(expert_id)+'.dense_h_to_4h_2.weight'] = \
                state_dict['model.layers.' + str(layer_id) + '.block_sparse_moe.experts.' + str(expert_id) + '.w3.weight']

            internal_state_dict['transformer.layers.' + str(layer_id) + '.mlp.megatron_moe.experts.megatron_experts.' + str(expert_id)+'.dense_4h_to_h.weight'] = state_dict[
                'model.layers.' + str(layer_id) + '.block_sparse_moe.experts.'+str(expert_id) +'.w2.weight']

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
                if op_name.startswith("input_layernorm") and weight_or_bias == "weight":
                    out_name = "self_attention.linear_qkv"
                    layer_name = f"layers.{layer}.{out_name}.layer_norm_weight"

                elif op_name.startswith("post_attention_layernorm") and weight_or_bias == "weight":
                    out_name = "pre_mlp_layernorm"
                    layer_name = f"layers.{layer}.{out_name}.{weight_or_bias}"

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

                # skip
                else:
                    continue

                if op_name + "." + weight_or_bias in tensor_parallel_params:
                    dim = 1 if op_name + "." + weight_or_bias in column_split_tensor_parallel_params else 0
                    params = torch.chunk(params, args.target_tensor_model_parallel_size, dim=dim)

                for i in range(args.target_tensor_model_parallel_size):
                    params_dict = get_element_from_dict_by_path(output_state_dict[i], "model")
                    params_dict["decoder." + layer_name] = (
                        params[i].clone() if (op_name + "." + weight_or_bias in tensor_parallel_params) else params.clone()
                    )

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
                    params_dict[dense_h_to_4h_name] =\
                        torch.cat([dense_h_to_4h_1_weight, dense_h_to_4h_2_weight], dim=0)

                self_attn_query_name = f"decoder.layers.{layer}.self_attn.query.weight"
                query_weight = params_dict[self_attn_query_name]
                del params_dict[self_attn_query_name]
                self_attn_kv_name = f"decoder.layers.{layer}.self_attn.key_value.weight"
                kv_weight = params_dict[self_attn_kv_name]
                del params_dict[self_attn_kv_name]

                # torch.Size([8 512, 4096])
                group_query_weight = query_weight.view(num_groups // args.target_tensor_model_parallel_size, num_heads // num_groups * hidden_size_per_head, hidden_size)
                # torch.Size(8, 256, 4096])
                group_kv_weight = kv_weight.view(num_groups // args.target_tensor_model_parallel_size, 2 * hidden_size_per_head, hidden_size)
                group_qkv_weight = torch.cat([group_query_weight, group_kv_weight], dim=1)
                params_dict["decoder." + f"layers.{layer}.self_attention.linear_qkv.weight"] =\
                    group_qkv_weight.view(-1, hidden_size)

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
            checkpoint_dir = (
                f"mp_rank_{tp_rank:02d}"
                if args.target_pipeline_model_parallel_size == 1
                else f"mp_rank_{tp_rank:02d}_{pp_rank:03d}"
            )

            save_dir = os.path.join(release_dir, checkpoint_dir)
            os.makedirs(save_dir, exist_ok=True)
            checkpoint_name = "model_optim_rng.pt"
            checkpoint_path = os.path.join(save_dir, checkpoint_name)
            torch.save(output_state_dict[tp_rank], checkpoint_path)

def main():

    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    convert_checkpoint_from_transformers_to_megatron(args)

if __name__ == "__main__":
    main()

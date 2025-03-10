# Copyright (c) 2025 Alibaba PAI Team.
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

import gc
import safetensors.torch
import sys
import os
import re
import torch
from collections import defaultdict
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from megatron.training.initialize import initialize_megatron
from megatron.training import get_args
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, read_metadata

path_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.append(os.path.join(path_dir, "examples"))
from deepseek_v3.pretrain_deepseek import model_provider
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

    parser.add_argument(
        "--target-tensor-model-parallel-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--target-pipeline-model-parallel-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--target-expert-model-parallel-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--target-decoder-first-pipeline-num-layers",
        type=int,
        default=None
    )

    parser.add_argument(
        "--hf-ckpt-path",
        type=str
    )

    parser.add_argument(
        "--save-safetensors",
        action='store_false',
    )

    return parser

def load_megatron_model(args):
    os.makedirs(args.save, exist_ok=True)
    os.system("cp -rf " + args.hf_ckpt_path + "/*config.json " + args.save)
    os.system("cp -rf " + args.hf_ckpt_path + "/tokenizer* " + args.save)
    os.system("cp -rf " + args.hf_ckpt_path + "/*.py " + args.save)

    os.system("cp -rf " + args.hf_ckpt_path + "/*config.json " + args.load)
    os.system("cp -rf " + args.hf_ckpt_path + "/tokenizer* " + args.load)
    os.system("cp -rf " + args.hf_ckpt_path + "/*.py " + args.load)

    model = model_provider()

    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size

    if args.num_experts is not None:
        args.expert_model_parallel_size = args.target_expert_model_parallel_size

    if args.tensor_model_parallel_size > 1:
        args.sequence_parallel = True

    model_path = args.load
    tracker_filename = get_checkpoint_tracker_filename(model_path)
    iteration, release = read_metadata(tracker_filename)
    q_head_dim = args.qk_head_dim + args.qk_pos_emb_head_dim
    group_per_split = args.num_attention_heads // args.tensor_model_parallel_size
    if args.num_experts is not None:
        pattern = r'weight(\d+)'
        num_local_experts = args.num_experts // args.expert_model_parallel_size
    state_dict = {}
    mid_state = defaultdict(list)

    if (
        args.tensor_model_parallel_size >= 1
        and args.pipeline_model_parallel_size >= 1
        and args.expert_model_parallel_size >= 1
        and args.num_experts % args.expert_model_parallel_size == 0
    ):
        #assert args.num_layers % args.pipeline_model_parallel_size == 0
        if args.target_decoder_first_pipeline_num_layers is not None:
            remained_layers = args.num_layers - args.target_decoder_first_pipeline_num_layers
            remained_stages = args.pipeline_model_parallel_size - 1
            assert remained_layers % remained_stages == 0
            pp_layers_per_stage = [args.target_decoder_first_pipeline_num_layers] +([remained_layers // remained_stages] * remained_stages)
        else:
            pp_layers_per_stage = [args.num_layers // args.pipeline_model_parallel_size] * args.pipeline_model_parallel_size

        layers_to_copy = {}
        for tp_rank in range(args.tensor_model_parallel_size):
            for ep_rank in range(args.expert_model_parallel_size):
                for pp_rank in range(args.pipeline_model_parallel_size):
                    layer_offset = sum(pp_layers_per_stage[:pp_rank])
                    for layer in range(pp_layers_per_stage[pp_rank]):
                        pp_layer_id = layer + layer_offset
                        layers_to_copy[(pp_rank, layer)] = pp_layer_id

                    if args.expert_model_parallel_size > 1:
                        checkpoint_name = get_checkpoint_name(model_path, iteration, release, True, tp_rank, pp_rank, True,
                                                              ep_rank)
                    elif args.expert_model_parallel_size == 1:
                        checkpoint_name = get_checkpoint_name(model_path, iteration, release, True, tp_rank, pp_rank,
                                                              False)
                    print(f'load {checkpoint_name}')
                    split_state = torch.load(checkpoint_name, map_location="cpu", weights_only=False)['model']
                    for k, v in split_state.items():
                        try:
                            if 'experts' in k:
                                local_expert_rank = int(re.findall(pattern, k)[0])
                                expert_rank = local_expert_rank + num_local_experts * ep_rank
                                k = k.replace(f'experts.{local_expert_rank}', f'experts.{expert_rank}')
                            pattern = re.compile(r'\d+')
                            res = pattern.findall(k)
                            tgt = re.sub(r"decoder.layers.\d+", "decoder.layers." + str(layers_to_copy[(pp_rank, int(res[0]))]), k)
                            if 'linear_proj' in k or 'linear_q_down_proj' in k or 'linear_q_up_proj'in k or 'linear_kv_up_proj' in k or 'linear_kv_down_proj' in k or\
                                    'decoder.layers.0.mlp.linear_fc1' in k or 'decoder.layers.1.mlp.linear_fc1' in k or 'decoder.layers.2.mlp.linear_fc1' in k or \
                                    'decoder.layers.0.mlp.linear_fc2' in k or 'decoder.layers.1.mlp.linear_fc2' in k or 'decoder.layers.2.mlp.linear_fc2' in k or \
                                    'shared_experts.linear_fc1' in k or 'shared_experts.linear_fc2' in k:
                                if ep_rank ==0:
                                    mid_state[tgt].append(v)
                            else:
                                mid_state[tgt].append(v)
                        except:
                            if "word_embeddings" in k:
                                if ep_rank ==0 and pp_rank == 0:
                                    mid_state[k].append(v)
                            elif "output_layer" in k or "final_layernorm" in k:
                                if ep_rank ==0 and pp_rank == args.pipeline_model_parallel_size - 1:
                                    mid_state[k].append(v)
                            else:
                                raise ValueError(f"{k} is missing! ")

        for k, v in mid_state.items():
            if 'extra_state' in k:
                continue
            if not isinstance(v[0], torch.Tensor) or 'router' in k or 'gate' in k:
                target_v = v[0]
            elif 'input_layernorm' in k:
                target_v = v[0]
            elif 'pre_mlp_layernorm' in k:
                target_v = v[0]
            elif 'word_embeddings' in k or 'output_layer' in k or 'final_layernorm' in k:
                target_v = torch.cat(v, dim=0)
            elif 'linear_q_down_proj' in k:
                target_v = torch.cat(v, dim=0)
            elif 'linear_q_up_proj' in k and 'layer_norm_weight' not in k:
                target_v = torch.cat(v, dim=0)
            elif 'linear_q_up_proj.layer_norm_weight' in k:
                target_v = torch.cat(v, dim=0)
            elif 'linear_kv_down_proj' in k:
                target_v = torch.cat(v, dim=0)
            elif 'linear_kv_up_proj' in k and 'layer_norm_weight' not in k:
                viewed = [x.view(group_per_split, -1, q_head_dim - args.qk_pos_emb_head_dim + args.v_head_dim, args.kv_lora_rank) for x in v]
                target_v = torch.cat(viewed, dim=0).view(-1, args.kv_lora_rank)
            elif 'linear_kv_up_proj.layer_norm_weight' in k:
                target_v = torch.cat(v, dim=0)
            elif 'linear_proj' in k:
                target_v = torch.cat(v, dim=1)
            elif 'linear_fc1' in k:
                viewed = [x.view(2, -1, args.hidden_size) for x in v]
                target_v = torch.cat(viewed, dim=1).view(-1, args.hidden_size)
            elif 'linear_fc2' in k:
                target_v = torch.cat(v, dim=1)
            else:
                raise ValueError(f"{k} is missing!")
            state_dict[k] = target_v

    else:
        raise ValueError('not support yet')

    model.load_state_dict(state_dict, strict=False)
    return model


def convert_checkpoint_from_megatron_to_transformers(mgmodel, hfmodel, args):

    if args.fp16:
        mgmodel = mgmodel.half()
    elif args.bf16:
        mgmodel = mgmodel.bfloat16()

    with torch.no_grad():
        hfmodel.model.embed_tokens.weight.copy_(mgmodel.embedding.word_embeddings.weight)
        for layer_idx, (mglayer, hflayer) in enumerate(zip(mgmodel.decoder.layers, hfmodel.model.layers)):
            print(layer_idx)
            hflayer.input_layernorm.weight.copy_(mglayer.input_layernorm.weight)
            hflayer.post_attention_layernorm.weight.copy_(mglayer.pre_mlp_layernorm.weight)

            hflayer.self_attn.q_a_proj.weight.copy_(mglayer.self_attention.linear_q_down_proj.weight)
            hflayer.self_attn.q_b_proj.weight.copy_(mglayer.self_attention.linear_q_up_proj.weight)
            hflayer.self_attn.q_a_layernorm.weight.copy_(mglayer.self_attention.linear_q_up_proj.layer_norm_weight)


            hflayer.self_attn.kv_a_proj_with_mqa.weight.copy_(mglayer.self_attention.linear_kv_down_proj.weight)
            hflayer.self_attn.kv_b_proj.weight.copy_(mglayer.self_attention.linear_kv_up_proj.weight)
            hflayer.self_attn.kv_a_layernorm.weight.copy_(mglayer.self_attention.linear_kv_up_proj.layer_norm_weight)
            hflayer.self_attn.o_proj.weight.copy_(mglayer.self_attention.linear_proj.weight)

            if layer_idx < 3:
                gate_weight, up_weight = torch.split(mglayer.mlp.linear_fc1.weight, split_size_or_sections=args.ffn_hidden_size)
                hflayer.mlp.gate_proj.weight.copy_(gate_weight)
                hflayer.mlp.up_proj.weight.copy_(up_weight)
                hflayer.mlp.down_proj.weight.copy_(mglayer.mlp.linear_fc2.weight)

            else:
                hflayer.mlp.gate.weight.copy_(mglayer.mlp.router.weight)

                for i, hfexpert in enumerate(hflayer.mlp.experts):
                    linear_fc1_weighti = getattr(mglayer.mlp.experts.linear_fc1, 'weight' + str(i))
                    gate_weight, up_weight = torch.split(linear_fc1_weighti,
                                                         split_size_or_sections=args.moe_ffn_hidden_size)
                    hfexpert.gate_proj.weight.copy_(gate_weight)
                    hfexpert.up_proj.weight.copy_(up_weight)
                    linear_fc2_weighti = getattr(mglayer.mlp.experts.linear_fc2, 'weight' + str(i))
                    hfexpert.down_proj.weight.copy_(linear_fc2_weighti)

                shared_expert_gate_weight, shared_expert_up_weight = \
                    torch.split(mglayer.mlp.shared_experts.linear_fc1.weight,
                                split_size_or_sections=args.moe_shared_expert_intermediate_size)
                hflayer.mlp.shared_experts.gate_proj.weight.copy_(shared_expert_gate_weight)
                hflayer.mlp.shared_experts.up_proj.weight.copy_(shared_expert_up_weight)
                hflayer.mlp.shared_experts.down_proj.weight.copy_(mglayer.mlp.shared_experts.linear_fc2.weight)

        hfmodel.model.norm.weight.copy_(mgmodel.decoder.final_layernorm.weight)
        hfmodel.lm_head.weight.copy_(mgmodel.output_layer.weight)


def convert_checkpoint_from_transformers_to_megatron(hfmodel, mgmodel, args):

    if args.fp16:
        mgmodel = mgmodel.half()
    elif args.bf16:
        mgmodel = mgmodel.bfloat16()
    if args.use_multi_token_prediction:
        file_path_160 = args.load+"/model-00160-of-000163.safetensors"
        file_path_161 = args.load + "/model-00161-of-000163.safetensors"
        file_path_162 = args.load + "/model-00162-of-000163.safetensors"
        file_path_163 = args.load + "/model-00163-of-000163.safetensors"
        with open(file_path_160, "rb") as f160, open(file_path_161, "rb") as f161,\
                open(file_path_162, "rb") as f162, open(file_path_163, "rb") as f163:
            b160 = f160.read()
            state_dict_160 = safetensors.torch.load(b160)
            b161 = f161.read()
            state_dict_161 = safetensors.torch.load(b161)
            b162 = f162.read()
            state_dict_162 = safetensors.torch.load(b162)
            b163 = f163.read()
            state_dict_163 = safetensors.torch.load(b163)

        def mergedict(*args):
            output = {}
            for arg in args:
                output.update(arg)
            return output

        mtp_dict = mergedict(state_dict_160, state_dict_161, state_dict_162, state_dict_163)
        with torch.no_grad():
            mgmodel.mtp_embedding.word_embeddings.weight.copy_(mtp_dict["model.layers.61.embed_tokens.weight"])
            mgmodel.mtp_predictor.mtp_modules[0].norm1.weight.copy_(mtp_dict['model.layers.61.enorm.weight'])
            mgmodel.mtp_predictor.mtp_modules[0].norm2.weight.copy_(mtp_dict['model.layers.61.hnorm.weight'])
            mgmodel.mtp_predictor.mtp_modules[0].linear_proj.weight.copy_(mtp_dict['model.layers.61.eh_proj.weight'])
            mtplayer = mgmodel.mtp_predictor.mtp_modules[0].decoder.layers[0]
            mtplayer.input_layernorm.weight.copy_(mtp_dict["model.layers.61.input_layernorm.weight"])
            mtplayer.pre_mlp_layernorm.weight.copy_(mtp_dict["model.layers.61.post_attention_layernorm.weight"])

            mtplayer.self_attention.linear_q_down_proj.weight.copy_(mtp_dict["model.layers.61.self_attn.q_a_proj.weight"])
            mtplayer.self_attention.linear_q_up_proj.weight.copy_(mtp_dict["model.layers.61.self_attn.q_b_proj.weight"])
            mtplayer.self_attention.linear_q_up_proj.layer_norm_weight.copy_(mtp_dict["model.layers.61.self_attn.q_a_layernorm.weight"])

            mtplayer.self_attention.linear_kv_down_proj.weight.copy_(mtp_dict["model.layers.61.self_attn.kv_a_proj_with_mqa.weight"])
            mtplayer.self_attention.linear_kv_up_proj.weight.copy_(mtp_dict["model.layers.61.self_attn.kv_b_proj.weight"])
            mtplayer.self_attention.linear_kv_up_proj.layer_norm_weight.copy_(mtp_dict["model.layers.61.self_attn.kv_a_layernorm.weight"])
            mtplayer.self_attention.linear_proj.weight.copy_(mtp_dict["model.layers.61.self_attn.o_proj.weight"])
            mtplayer.mlp.router.weight.copy_(mtp_dict["model.layers.61.mlp.gate.weight"])
            mtplayer.mlp.router.expert_bias.copy_(mtp_dict["model.layers.61.mlp.gate.e_score_correction_bias"])
            for i in range(args.num_experts):
                fc1_weight = torch.cat([mtp_dict["model.layers.61.mlp.experts."+str(i)+".gate_proj.weight"],
                                        mtp_dict["model.layers.61.mlp.experts."+str(i)+".up_proj.weight"]])
                linear_fc1_weighti = getattr(mtplayer.mlp.experts.linear_fc1, 'weight' + str(i))
                linear_fc1_weighti.copy_(fc1_weight)
                linear_fc2_weighti = getattr(mtplayer.mlp.experts.linear_fc2, 'weight' + str(i))
                linear_fc2_weighti.copy_(mtp_dict["model.layers.61.mlp.experts."+str(i)+".down_proj.weight"])

            shared_fc1_weight = torch.cat(
                [mtp_dict["model.layers.61.mlp.shared_experts.gate_proj.weight"],
                 mtp_dict["model.layers.61.mlp.shared_experts.up_proj.weight"]])
            mtplayer.mlp.shared_experts.linear_fc1.weight.copy_(shared_fc1_weight)
            mtplayer.mlp.shared_experts.linear_fc2.weight.copy_(mtp_dict["model.layers.61.mlp.shared_experts.down_proj.weight"])
            mgmodel.output_layer.weight.copy_(hfmodel.lm_head.weight)

    with torch.no_grad():
        mgmodel.embedding.word_embeddings.weight.copy_(hfmodel.model.embed_tokens.weight)

        for layer_idx, (mglayer, hflayer) in enumerate(zip(mgmodel.decoder.layers, hfmodel.model.layers)):
            print(layer_idx)
            mglayer.input_layernorm.weight.copy_(hflayer.input_layernorm.weight)
            mglayer.pre_mlp_layernorm.weight.copy_(hflayer.post_attention_layernorm.weight)

            mglayer.self_attention.linear_q_down_proj.weight.copy_(hflayer.self_attn.q_a_proj.weight)
            mglayer.self_attention.linear_q_up_proj.weight.copy_(hflayer.self_attn.q_b_proj.weight)
            mglayer.self_attention.linear_q_up_proj.layer_norm_weight.copy_(hflayer.self_attn.q_a_layernorm.weight)

            mglayer.self_attention.linear_kv_down_proj.weight.copy_(hflayer.self_attn.kv_a_proj_with_mqa.weight)
            mglayer.self_attention.linear_kv_up_proj.weight.copy_(hflayer.self_attn.kv_b_proj.weight)
            mglayer.self_attention.linear_kv_up_proj.layer_norm_weight.copy_(hflayer.self_attn.kv_a_layernorm.weight)
            mglayer.self_attention.linear_proj.weight.copy_(hflayer.self_attn.o_proj.weight)
            if layer_idx < 3:
                mglayer.mlp.linear_fc1.weight.copy_(
                    torch.cat([hflayer.mlp.gate_proj.weight, hflayer.mlp.up_proj.weight]))
                mglayer.mlp.linear_fc2.weight.copy_(hflayer.mlp.down_proj.weight)
            else:
                mglayer.mlp.router.weight.copy_(hflayer.mlp.gate.weight)
                mglayer.mlp.router.expert_bias.copy_(hflayer.mlp.gate.e_score_correction_bias)
                for i, hf_expert in enumerate(hflayer.mlp.experts):
                    fc1_weight = torch.cat([hf_expert.gate_proj.weight, hf_expert.up_proj.weight])
                    linear_fc1_weighti = getattr(mglayer.mlp.experts.linear_fc1, 'weight' + str(i))
                    linear_fc1_weighti.copy_(fc1_weight)
                    linear_fc2_weighti = getattr(mglayer.mlp.experts.linear_fc2, 'weight' + str(i))
                    linear_fc2_weighti.copy_(hf_expert.down_proj.weight)

                shared_fc1_weight = torch.cat(
                    [hflayer.mlp.shared_experts.gate_proj.weight, hflayer.mlp.shared_experts.up_proj.weight])
                mglayer.mlp.shared_experts.linear_fc1.weight.copy_(shared_fc1_weight)
                mglayer.mlp.shared_experts.linear_fc2.weight.copy_(hflayer.mlp.shared_experts.down_proj.weight)

        mgmodel.decoder.final_layernorm.weight.copy_(hfmodel.model.norm.weight)
        if args.untie_embeddings_and_output_weights:
            mgmodel.output_layer.weight.copy_(hfmodel.lm_head.weight)


def check_layer(layers_to_copy, k):
    pattern = re.compile(r"decoder.layers.\d+")
    res = pattern.findall(k)
    return res and res[0] in layers_to_copy.keys()

def save_mgmodel(mgmodel, args):

    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size

    if args.num_experts is not None:
        args.expert_model_parallel_size = args.target_expert_model_parallel_size

    os.makedirs(args.save, exist_ok=True)
    os.system("cp -rf " + args.load + "/*config.json " + args.save)
    os.system("cp -rf " + args.load + "/tokenizer* " + args.save)

    tracker_filepath = os.path.join(args.save, 'latest_checkpointed_iteration.txt')
    with open(tracker_filepath, "w") as f:
        f.write("release")

    full_model = mgmodel.state_dict_for_save_checkpoint()
    for k in list(full_model.keys()):
        if full_model[k] is None and '_extra_state' not in k:
            full_model.pop(k)
            continue
        if '_extra_state' in k and isinstance(full_model[k], torch.Tensor):
            full_model[k] = None

    if args.num_experts is not None:
        pattern = r'weight(\d+)'
        num_local_experts = args.num_experts // args.expert_model_parallel_size if args.num_experts else 0

    if (
        args.tensor_model_parallel_size >= 1
        and args.pipeline_model_parallel_size > 1
        and args.num_experts % args.expert_model_parallel_size == 0
    ):

        if args.target_decoder_first_pipeline_num_layers is not None:
            remained_layers = args.num_layers - args.target_decoder_first_pipeline_num_layers
            remained_stages = args.pipeline_model_parallel_size - 1
            assert remained_layers % remained_stages == 0
            pp_layers_per_stage = [ args.target_decoder_first_pipeline_num_layers] +([remained_layers // remained_stages] * remained_stages)
        else:
            pp_layers_per_stage = [args.num_layers // args.pipeline_model_parallel_size] * args.pipeline_model_parallel_size

        for tp_rank in range(args.tensor_model_parallel_size):
            for ep_rank in range(tp_rank, args.expert_model_parallel_size, args.tensor_model_parallel_size):
                for pp_rank in range(args.pipeline_model_parallel_size):
                    model_split = {}
                    layer_offset = sum(pp_layers_per_stage[:pp_rank])
                    layers_to_copy = {}
                    for layer in range(pp_layers_per_stage[pp_rank]):
                        pp_layer_id = layer + layer_offset
                        layers_to_copy[f"decoder.layers.{pp_layer_id}"] = layer
                    if args.expert_model_parallel_size > 1:
                        checkpoint_name = get_checkpoint_name(args.save, 0, True, True, tp_rank, pp_rank, True, ep_rank)
                    elif args.expert_model_parallel_size == 1:
                        checkpoint_name = get_checkpoint_name(args.save, 0, True, True, tp_rank, pp_rank, False)
                    print(f'tensor_parallel & pipeline_parallel & expert_parallel, save model to {checkpoint_name}')
                    for k, v in full_model.items():
                        if check_layer(layers_to_copy, k):
                            layer_pattern = re.compile(r'\d+')
                            res = layer_pattern.findall(k)
                            k = re.sub(r"decoder.layers.\d+", "decoder.layers." + str(layers_to_copy["decoder.layers." + res[0]]), k)
                        elif not ("word_embeddings" in k or "output_layer" in k or "final_layernorm" in k or "mtp_modules" in k):
                            continue

                        if not isinstance(v, torch.Tensor):
                            target_v = v
                        elif 'linear_q_down_proj' in k or 'linear_kv_down_proj' in k:
                            seg = v.shape[0] // args.tensor_model_parallel_size
                            target_v = v[seg * tp_rank: seg * (tp_rank + 1)]
                            #target_v = v
                        elif 'linear_q_up_proj.layer_norm_weight' in k or 'linear_kv_up_proj.layer_norm_weight' in k:
                            #seg = v.shape[0] // args.tensor_model_parallel_size
                            #target_v = v[seg * tp_rank: seg * (tp_rank + 1)]
                            target_v = v
                        elif 'linear_q_up_proj' in k and 'layer_norm_weight' not in k:
                            seg = v.shape[0] // args.tensor_model_parallel_size
                            target_v = v[seg * tp_rank:seg * (tp_rank + 1)]
                        elif 'linear_kv_up_proj' in k and 'layer_norm_weight' not in k:
                            seg = v.shape[0] // args.tensor_model_parallel_size
                            target_v = v[seg * tp_rank:seg * (tp_rank + 1)]
                        elif 'linear_proj' in k and 'mtp_predictor' not in k:
                            seg = v.shape[1] // args.tensor_model_parallel_size
                            target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]
                        elif 'mtp_predictor.mtp_modules.0.decoder.layers.0.self_attention.linear_proj' in k:
                            seg = v.shape[1] // args.tensor_model_parallel_size
                            target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]
                        elif 'decoder.layers.0.mlp.linear_fc2' in k or 'decoder.layers.1.mlp.linear_fc2' in k or 'decoder.layers.2.mlp.linear_fc2' in k:
                            seg = v.shape[1] // args.tensor_model_parallel_size
                            target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]
                        elif 'decoder.layers.0.mlp.linear_fc1' in k or 'decoder.layers.1.mlp.linear_fc1'in k or 'decoder.layers.2.mlp.linear_fc1' in k:
                            seg = args.ffn_hidden_size // args.tensor_model_parallel_size
                            viewed = v.view(-1, args.ffn_hidden_size, args.hidden_size)
                            target_v = viewed[:, seg * tp_rank: seg * (tp_rank + 1), :].reshape(-1, args.hidden_size)
                        elif 'experts' in k and 'shared_experts' not in k:
                            expert_rank = int(re.findall(pattern, k)[0])
                            if expert_rank // num_local_experts != ep_rank:
                                continue
                            expert_local_rank = expert_rank % num_local_experts
                            k = k.replace(f'weight{expert_rank}', f'weight{expert_local_rank}')
                            """
                            if 'linear_fc1' in k:
                                viewed = v.view(-1, args.moe_ffn_hidden_size, args.hidden_size)
                                seg = args.moe_ffn_hidden_size // args.tensor_model_parallel_size
                                target_v = viewed[:, seg * tp_rank: seg * (tp_rank + 1), :].reshape(-1, args.hidden_size)
                            elif 'linear_fc2' in k:
                                seg = v.shape[1] // args.tensor_model_parallel_size
                                target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]
                            """
                            target_v = v
                        elif 'shared_experts' in k and 'gate' not in k:
                            if 'linear_fc1' in k:
                                viewed = v.view(-1, args.moe_shared_expert_intermediate_size,
                                                args.hidden_size)
                                seg = args.moe_shared_expert_intermediate_size // args.tensor_model_parallel_size
                                target_v = viewed[:, seg * tp_rank: seg * (tp_rank + 1), :].reshape(-1, args.hidden_size)
                            elif 'linear_fc2' in k:
                                seg = v.shape[1] // args.tensor_model_parallel_size
                                target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]
                        elif "word_embeddings" in k or "output_layer" in k:
                            seg = v.shape[0] // args.tensor_model_parallel_size
                            target_v = v[seg * tp_rank: seg * (tp_rank + 1)]
                        else:
                            target_v = v

                        if "embedding.word_embeddings" in k and "mtp_embedding" not in k:
                            if pp_rank == 0:
                                model_split[k] = target_v
                        elif "mtp_embedding.word_embeddings" in k:
                            if pp_rank == args.pipeline_model_parallel_size - 1:
                                model_split[k] = target_v
                        elif "output_layer" in k or "final_layernorm" in k or "mtp_modules" in k:
                            if pp_rank == args.pipeline_model_parallel_size - 1:
                                model_split[k] = target_v
                        else:
                            model_split[k] = target_v
                    save_state_dict(args, [model_split], checkpoint_name)

    else:
        raise ValueError('Something is wrong, please check your tp/pp/ep size')

    print(f'megatron model is save to {args.save}')

def add_extra_args(parser):
    parser = get_patch_args(parser)
    parser = add_model_args(parser)
    return parser

def main():
    initialize_megatron(extra_args_provider=add_extra_args)
    args = get_args()

    if args.convert_checkpoint_from_megatron_to_transformers:
        config = AutoConfig.from_pretrained(args.hf_ckpt_path, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_pretrained(args.hf_ckpt_path, trust_remote_code=True, torch_dtype=config.torch_dtype)
        mg_model = load_megatron_model(args)
        convert_checkpoint_from_megatron_to_transformers(mg_model, hf_model, args)
        save_hfmodel(args, hf_model)
    else:
        config = AutoConfig.from_pretrained(args.load, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_pretrained(args.load, trust_remote_code=True, torch_dtype=config.torch_dtype)
        mg_model = model_provider()
        convert_checkpoint_from_transformers_to_megatron(hf_model, mg_model, args)
        del hf_model
        gc.collect()
        save_mgmodel(mg_model, args)

if __name__ == "__main__":
    main()
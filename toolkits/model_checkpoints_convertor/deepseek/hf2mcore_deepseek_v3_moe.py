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
import json
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

from safetensors import safe_open
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, read_metadata

path_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.append(os.path.join(path_dir, "examples"))
from deepseek_v3.pretrain_deepseek import model_provider
from megatron_patch.arguments import get_patch_args

from toolkits.model_checkpoints_convertor.utils import (
    save_state_dict,
    save_hfmodel,
    safe_copy
)
import math

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

def generate_rank_group(
    tensor_model_parallel_size,
    expert_tensor_parallel_size,
    expert_model_parallel_size,
    pipeline_model_parallel_size
):
    """
        This function attempts to generate group rank on the minimal practicable world_size.
        Support Decoder-Only model currently.
    """
    tp, etp, ep, pp = (
        tensor_model_parallel_size,
        expert_tensor_parallel_size,
        expert_model_parallel_size,
        pipeline_model_parallel_size
    )
    minimal_worldsize = pp * math.lcm(tp, etp * ep)
    print(f"The given parallel config should be run on at least {minimal_worldsize} cards")
    dp = minimal_worldsize // (pp * tp)
    edp = minimal_worldsize // (pp * ep * etp)
    # NOTE: If user want to scale up cp_size, he should downscale 
    # dp_size or scale up world_size, i.e., edp_size
    cp = 1
    
    # TODO: support other orders
    order="tp-cp-ep-dp-pp"
    # In training:
    # Dense: 
    # global_rank = tp_rank + cp_rank * tp_size + dp_rank * cp_size * tp_size + pp_rank * dp_size * cp_size * tp_size
    # MoE:
    # global_rank = etp_rank + ep_rank * etp_size + edp_rank * ep_size * etp_size + pp_rank * edp_size * ep_size * etp_size

    # In ckpt loading, each rank will load a checkpoint according to its (tp_rank, pp_rank, ep_rank)
    # Thus, (tp_rank, ep_rank) should map to a unique etp_rank
    rank_mappings = dict()
    local_ids = []
    for global_rank in range(minimal_worldsize):
        tp_rank = global_rank % tp
        etp_rank = global_rank % etp
        ep_rank = (global_rank // etp) % ep
        pp_rank = global_rank // (dp * tp)

        if (tp_rank, ep_rank) not in rank_mappings:
            rank_mappings[(tp_rank, ep_rank)] = etp_rank
        
        if rank_mappings[(tp_rank, ep_rank)] != etp_rank:
            raise ValueError("The legacy checkpoint format cannot support this parallel config.")

        local_ids.append((tp_rank, etp_rank, ep_rank, pp_rank))
    return local_ids

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
        "--target-expert-tensor-parallel-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--target-decoder-first-pipeline-num-layers",
        type=int,
        default=None
    )

    parser.add_argument(
        "--target-decoder-last-pipeline-num-layers",
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
    args.expert_model_parallel_size = args.target_expert_model_parallel_size
    args.expert_tensor_parallel_size = args.target_expert_tensor_parallel_size

    if args.tensor_model_parallel_size > 1:
        args.sequence_parallel = True

    model_path = args.load
    tracker_filename = get_checkpoint_tracker_filename(model_path)
    iteration, release = read_metadata(tracker_filename)
    q_head_dim = args.qk_head_dim + args.qk_pos_emb_head_dim
    group_per_split = args.num_attention_heads // args.tensor_model_parallel_size
    if args.num_experts is not None:
        if args.moe_grouped_gemm == True:
            ep_pattern = r'weight(\d+)'
        else:
            ep_pattern = r'local_experts\.(\d+)\.'
        num_local_experts = args.num_experts // args.expert_model_parallel_size
    state_dict = {}
    mid_state = defaultdict(list)

    assert args.num_experts % args.expert_model_parallel_size == 0, "The amount of experts should be evenly split across EP ranks."

    pp_layers_per_stage = []
    remained_layers = args.num_layers
    remained_stages = args.pipeline_model_parallel_size
    if args.target_decoder_first_pipeline_num_layers is not None:
        remained_layers -= args.target_decoder_first_pipeline_num_layers
        remained_stages -= 1
        pp_layers_per_stage = [args.target_decoder_first_pipeline_num_layers]
    
    if args.target_decoder_last_pipeline_num_layers is not None:
        remained_layers -= args.target_decoder_last_pipeline_num_layers
        remained_stages -= 1

    assert remained_layers % remained_stages == 0
    pp_layers_per_stage.extend([remained_layers // remained_stages] * remained_stages)
    if args.target_decoder_last_pipeline_num_layers is not None:
        pp_layers_per_stage.append(args.target_decoder_last_pipeline_num_layers)

    layers_to_copy = {}
    for (tp_rank, etp_rank, ep_rank, pp_rank) in generate_rank_group(
            args.tensor_model_parallel_size,
            args.expert_tensor_parallel_size,
            args.expert_model_parallel_size,
            args.pipeline_model_parallel_size
        ):
            layer_offset = sum(pp_layers_per_stage[:pp_rank])
            for layer in range(pp_layers_per_stage[pp_rank]):
                pp_layer_id = layer + layer_offset
                layers_to_copy[(pp_rank, layer)] = pp_layer_id

            checkpoint_name = get_checkpoint_name(
                model_path, 
                iteration, 
                release, 
                args.pipeline_model_parallel_size > 1, 
                tp_rank, 
                pp_rank, 
                args.expert_model_parallel_size > 1,
                ep_rank
            )

            print(f'load {checkpoint_name}')
            split_state = torch.load(checkpoint_name, map_location="cpu")['model']
            for k, v in split_state.items():
                if '_extra_state' in k:
                    continue
                try:
                    pattern = re.compile(r'\d+')
                    res = pattern.findall(k)
                    tgt_layer_id = layers_to_copy[(pp_rank, int(res[0]))]
                    tgt = re.sub(r"decoder.layers.\d+", "decoder.layers." + str(tgt_layer_id), k)

                    if 'experts' in k and 'shared_experts' not in k:
                        local_expert_rank = int(re.findall(ep_pattern, k)[0])
                        expert_rank = local_expert_rank + num_local_experts * ep_rank
                        tgt = tgt.replace(f'weight{local_expert_rank}', f'weight{expert_rank}')
                        if tgt not in mid_state:
                            mid_state[tgt] = [None] * args.target_expert_tensor_parallel_size
                        # NOTE: deduplicate MoE params by individual ETP
                        if mid_state[tgt][etp_rank] is not None:
                            # NOTE: Here we can add a check to ensure parameters saved by mcore are synchronized.
                            pass
                        mid_state[tgt][etp_rank] = v                       
                    else:
                        if tgt not in mid_state:
                            mid_state[tgt] = [None] * args.target_tensor_model_parallel_size
                        if mid_state[tgt][tp_rank] is not None:
                            # NOTE: Here we can add a check to ensure parameters saved by mcore are synchronized.
                            pass
                        mid_state[tgt][tp_rank] = v
                except:
                    if contains(k, ["word_embeddings", "output_layer"]):
                        if k not in mid_state:
                            mid_state[k] = [None] * args.target_tensor_model_parallel_size
                        mid_state[k][tp_rank] = v
                    elif "final_layernorm" in k:
                        mid_state[k] = [v]
                    else:
                        raise ValueError(f"{k} is missing! ")

    for k, v in mid_state.items():
        try:
            if 'extra_state' in k:
                continue
            if not isinstance(v[0], torch.Tensor):
                target_v = v[0]
            elif contains(k, ['router', 'gate', 'input_layernorm', 'pre_mlp_layernorm', 'enorm', 'hnorm', 'layer_norm_weight', 'final_layernorm']):
                target_v = v[0]
            elif contains(k, ['word_embeddings', 'output_layer', 'linear_q_down_proj', 'linear_q_up_proj', 'linear_kv_down_proj', 'linear_kv_up_proj.layer_norm_weight', 'eh_proj', 'linear_q_proj']):
                target_v = torch.cat(v, dim=0)
            elif 'linear_kv_up_proj' in k:
                viewed = [x.view(group_per_split, -1, q_head_dim - args.qk_pos_emb_head_dim + args.v_head_dim, args.kv_lora_rank) for x in v]
                target_v = torch.cat(viewed, dim=0).view(-1, args.kv_lora_rank)
            elif 'linear_proj' in k:
                target_v = torch.cat(v, dim=1)
            elif 'linear_fc1' in k:
                viewed = [x.view(2, -1, args.hidden_size) for x in v]
                target_v = torch.cat(viewed, dim=1).view(-1, args.hidden_size)
            elif 'linear_fc2' in k:
                target_v = torch.cat(v, dim=1)
            else:
                raise ValueError(f"{k} is missing!")
        except Exception as e:
            print(f"Failed on {k} with shape {[item.shape for item in v]}")
            raise e
        state_dict[k] = target_v

    missing, unexpected =  model.load_state_dict(state_dict, strict=False)
    return model


def convert_checkpoint_from_megatron_to_transformers(mgmodel, hfmodel, args):
    if args.fp16:
        mgmodel = mgmodel.half()
    elif args.bf16:
        mgmodel = mgmodel.bfloat16()

    mtp_dict = {}
    if args.mtp_num_layers and args.mtp_num_layers > 0:
        mtp_layer_idx = args.num_layers
        # word_embedding
        mtp_dict[f"model.layers.{mtp_layer_idx}.embed_tokens.weight"] = mgmodel.embedding.word_embeddings.weight
        # e/h norm and eh_proj
        mtp_dict[f'model.layers.{mtp_layer_idx}.enorm.weight'] = mgmodel.mtp.layers[0].enorm.weight
        mtp_dict[f'model.layers.{mtp_layer_idx}.hnorm.weight'] = mgmodel.mtp.layers[0].hnorm.weight
        mtp_dict[f'model.layers.{mtp_layer_idx}.eh_proj.weight'] = mgmodel.mtp.layers[0].eh_proj.weight

        # attention and mlp
        mtplayer = mgmodel.mtp.layers[0].transformer_layer
        mtp_dict[f"model.layers.{mtp_layer_idx}.input_layernorm.weight"] = mtplayer.input_layernorm.weight
        mtp_dict[f"model.layers.{mtp_layer_idx}.post_attention_layernorm.weight"] = mtplayer.pre_mlp_layernorm.weight
        mtp_dict[f"model.layers.{mtp_layer_idx}.self_attn.q_a_proj.weight"] = mtplayer.self_attention.linear_q_down_proj.weight
        mtp_dict[f"model.layers.{mtp_layer_idx}.self_attn.q_b_proj.weight"] = mtplayer.self_attention.linear_q_up_proj.weight
        mtp_dict[f"model.layers.{mtp_layer_idx}.self_attn.q_a_layernorm.weight"] = mtplayer.self_attention.linear_q_up_proj.layer_norm_weight
        mtp_dict[f"model.layers.{mtp_layer_idx}.self_attn.kv_a_proj_with_mqa.weight"] = mtplayer.self_attention.linear_kv_down_proj.weight
        mtp_dict[f"model.layers.{mtp_layer_idx}.self_attn.kv_b_proj.weight"] = mtplayer.self_attention.linear_kv_up_proj.weight
        mtp_dict[f"model.layers.{mtp_layer_idx}.self_attn.kv_a_layernorm.weight"] = mtplayer.self_attention.linear_kv_up_proj.layer_norm_weight
        mtp_dict[f"model.layers.{mtp_layer_idx}.self_attn.o_proj.weight"] = mtplayer.self_attention.linear_proj.weight
        mtp_dict[f"model.layers.{mtp_layer_idx}.mlp.gate.weight"] = mtplayer.mlp.router.weight
        # NOTE: the e_score_correction_bias in mcore model will be initialized with bfloat16 and \
        # recover to fp32 in the first forward. Convert to fp32 to suit huggingface impl.
        mtp_dict[f"model.layers.{mtp_layer_idx}.mlp.gate.e_score_correction_bias"] = mtplayer.mlp.router.expert_bias.float()

        for i in range(args.num_experts):
            (
                mtp_dict[f"model.layers.{mtp_layer_idx}.mlp.experts."+str(i)+".gate_proj.weight"],
                mtp_dict[f"model.layers.{mtp_layer_idx}.mlp.experts."+str(i)+".up_proj.weight"]
            ) = torch.chunk(getattr(mtplayer.mlp.experts.linear_fc1, 'weight' + str(i)), 2)
            mtp_dict[f"model.layers.{mtp_layer_idx}.mlp.experts."+str(i)+".down_proj.weight"] = getattr(mtplayer.mlp.experts.linear_fc2, 'weight' + str(i))
            
        (
            mtp_dict[f"model.layers.{mtp_layer_idx}.mlp.shared_experts.gate_proj.weight"], 
            mtp_dict[f"model.layers.{mtp_layer_idx}.mlp.shared_experts.up_proj.weight"]
        ) = torch.chunk(mtplayer.mlp.shared_experts.linear_fc1.weight, 2)
        mtp_dict[f"model.layers.{mtp_layer_idx}.mlp.shared_experts.down_proj.weight"] = mtplayer.mlp.shared_experts.linear_fc2.weight
        # output norm
        mtp_dict[f"model.layers.{mtp_layer_idx}.shared_head.norm.weight"] = mgmodel.mtp.layers[0].final_layernorm.weight
        # shared output head
        mtp_dict[f"model.layers.{mtp_layer_idx}.shared_head.head.weight"] = mgmodel.output_layer.weight

    hfmodel.model.embed_tokens.weight.copy_(mgmodel.embedding.word_embeddings.weight)
    for layer_idx, (mglayer, hflayer) in enumerate(zip(mgmodel.decoder.layers, hfmodel.model.layers)):
        print(layer_idx)
        hflayer.input_layernorm.weight.copy_(mglayer.input_layernorm.weight)

        if args.q_lora_rank is None:
            hflayer.self_attn.q_proj.weight.copy_(mglayer.self_attention.linear_q_proj.weight)
        else:
            hflayer.self_attn.q_a_proj.weight.copy_(mglayer.self_attention.linear_q_down_proj.weight)
            hflayer.self_attn.q_b_proj.weight.copy_(mglayer.self_attention.linear_q_up_proj.weight)
            hflayer.self_attn.q_a_layernorm.weight.copy_(mglayer.self_attention.linear_q_up_proj.layer_norm_weight)

        hflayer.self_attn.kv_a_proj_with_mqa.weight.copy_(mglayer.self_attention.linear_kv_down_proj.weight)
        hflayer.self_attn.kv_b_proj.weight.copy_(mglayer.self_attention.linear_kv_up_proj.weight)
        hflayer.self_attn.kv_a_layernorm.weight.copy_(mglayer.self_attention.linear_kv_up_proj.layer_norm_weight)
        hflayer.self_attn.o_proj.weight.copy_(mglayer.self_attention.linear_proj.weight)

        if not hasattr(mglayer.mlp, 'router'):
            hflayer.post_attention_layernorm.weight.copy_(mglayer.mlp.linear_fc1.layer_norm_weight)
            gate_weight, up_weight = torch.split(mglayer.mlp.linear_fc1.weight, split_size_or_sections=args.ffn_hidden_size)
            hflayer.mlp.gate_proj.weight.copy_(gate_weight)
            hflayer.mlp.up_proj.weight.copy_(up_weight)
            hflayer.mlp.down_proj.weight.copy_(mglayer.mlp.linear_fc2.weight)
        else:
            hflayer.mlp.gate.weight.copy_(mglayer.mlp.router.weight)
            safe_copy(mglayer.mlp.router.expert_bias, hflayer.mlp.gate.e_score_correction_bias, skip_dtype_assert=False)
            for i, hfexpert in enumerate(hflayer.mlp.experts):
                linear_fc1_weighti = getattr(mglayer.mlp.experts.linear_fc1, 'weight' + str(i))
                gate_weight, up_weight = torch.split(linear_fc1_weighti,
                                                        split_size_or_sections=args.moe_ffn_hidden_size)
                hfexpert.gate_proj.weight.copy_(gate_weight)
                hfexpert.up_proj.weight.copy_(up_weight)
                linear_fc2_weighti = getattr(mglayer.mlp.experts.linear_fc2, 'weight' + str(i))
                hfexpert.down_proj.weight.copy_(linear_fc2_weighti)

            hflayer.post_attention_layernorm.weight.copy_(mglayer.pre_mlp_layernorm.weight)
            shared_expert_gate_weight, shared_expert_up_weight = \
                torch.split(mglayer.mlp.shared_experts.linear_fc1.weight,
                            split_size_or_sections=args.moe_shared_expert_intermediate_size)
            hflayer.mlp.shared_experts.gate_proj.weight.copy_(shared_expert_gate_weight)
            hflayer.mlp.shared_experts.up_proj.weight.copy_(shared_expert_up_weight)
            hflayer.mlp.shared_experts.down_proj.weight.copy_(mglayer.mlp.shared_experts.linear_fc2.weight)

    hfmodel.model.norm.weight.copy_(mgmodel.decoder.final_layernorm.weight)
    hfmodel.lm_head.weight.copy_(mgmodel.output_layer.weight)

    state_dict = hfmodel.state_dict()
    state_dict.update(mtp_dict)
    return state_dict

def convert_checkpoint_from_transformers_to_megatron(hfmodel, mgmodel, args):
    if args.fp16:
        mgmodel = mgmodel.half()
    elif args.bf16:
        mgmodel = mgmodel.bfloat16()

    if args.mtp_num_layers and args.mtp_num_layers > 0:
        mtp_layer_idx = args.num_layers
        # collect weights of model.layers.{mtp_layer_idx} from index.json
        mtp_dict = {}
        index_file = os.path.join(args.load, 'model.safetensors.index.json')
        if not os.path.exists(index_file):
            raise FileNotFoundError("'model.safetensors.index.json' not exists, cannot convert MTP module..")
        with open(index_file, 'r') as f:
            index_data = json.load(f)["weight_map"]
        mtp_map = {k:v for k, v in index_data.items() if f'model.layers.{mtp_layer_idx}' in k}
        files = set(mtp_map.values())
        for file in files:
            with safe_open(os.path.join(args.load, file), framework="pt") as f:
                mtp_dict.update({k: f.get_tensor(k) for k in f.keys() if k in mtp_map})
        
        # NOTE: no-need to copy shared embedding
        # mgmodel.embedding.word_embeddings.weight.copy_(mtp_dict[f"model.layers.{mtp_layer_idx}.embed_tokens.weight"])
        # e/h norm and eh_proj
        mgmodel.mtp.layers[0].enorm.weight.copy_(mtp_dict[f'model.layers.{mtp_layer_idx}.enorm.weight'])
        mgmodel.mtp.layers[0].hnorm.weight.copy_(mtp_dict[f'model.layers.{mtp_layer_idx}.hnorm.weight'])
        mgmodel.mtp.layers[0].eh_proj.weight.copy_(mtp_dict[f'model.layers.{mtp_layer_idx}.eh_proj.weight'])
        # attention and mlp
        mtplayer = mgmodel.mtp.layers[0].transformer_layer
        mtplayer.input_layernorm.weight.copy_(mtp_dict[f"model.layers.{mtp_layer_idx}.input_layernorm.weight"])
        mtplayer.pre_mlp_layernorm.weight.copy_(mtp_dict[f"model.layers.{mtp_layer_idx}.post_attention_layernorm.weight"])
        mtplayer.self_attention.linear_q_down_proj.weight.copy_(mtp_dict[f"model.layers.{mtp_layer_idx}.self_attn.q_a_proj.weight"])
        mtplayer.self_attention.linear_q_up_proj.weight.copy_(mtp_dict[f"model.layers.{mtp_layer_idx}.self_attn.q_b_proj.weight"])
        mtplayer.self_attention.linear_q_up_proj.layer_norm_weight.copy_(mtp_dict[f"model.layers.{mtp_layer_idx}.self_attn.q_a_layernorm.weight"])
        mtplayer.self_attention.linear_kv_down_proj.weight.copy_(mtp_dict[f"model.layers.{mtp_layer_idx}.self_attn.kv_a_proj_with_mqa.weight"])
        mtplayer.self_attention.linear_kv_up_proj.weight.copy_(mtp_dict[f"model.layers.{mtp_layer_idx}.self_attn.kv_b_proj.weight"])
        mtplayer.self_attention.linear_kv_up_proj.layer_norm_weight.copy_(mtp_dict[f"model.layers.{mtp_layer_idx}.self_attn.kv_a_layernorm.weight"])
        mtplayer.self_attention.linear_proj.weight.copy_(mtp_dict[f"model.layers.{mtp_layer_idx}.self_attn.o_proj.weight"])
        mtplayer.mlp.router.weight.copy_(mtp_dict[f"model.layers.{mtp_layer_idx}.mlp.gate.weight"])
        mtplayer.mlp.router.expert_bias.copy_(mtp_dict[f"model.layers.{mtp_layer_idx}.mlp.gate.e_score_correction_bias"])
        if args.moe_grouped_gemm == True:
            for i in range(args.num_experts):
                fc1_weight = torch.cat([mtp_dict[f"model.layers.{mtp_layer_idx}.mlp.experts."+str(i)+".gate_proj.weight"],
                                        mtp_dict[f"model.layers.{mtp_layer_idx}.mlp.experts."+str(i)+".up_proj.weight"]])
                linear_fc1_weighti = getattr(mtplayer.mlp.experts.linear_fc1, 'weight' + str(i))
                linear_fc1_weighti.copy_(fc1_weight)
                linear_fc2_weighti = getattr(mtplayer.mlp.experts.linear_fc2, 'weight' + str(i))
                linear_fc2_weighti.copy_(mtp_dict[f"model.layers.{mtp_layer_idx}.mlp.experts."+str(i)+".down_proj.weight"])
        else:
            for i in range(args.num_experts):
                expert = mtplayer.mlp.experts.local_experts[i]
                fc1_weight = torch.cat(
                    [mtp_dict[f"model.layers.{mtp_layer_idx}.mlp.experts." + str(i) + ".gate_proj.weight"],
                     mtp_dict[f"model.layers.{mtp_layer_idx}.mlp.experts." + str(i) + ".up_proj.weight"]])

                expert.linear_fc1.weight.copy_(fc1_weight)
                expert.linear_fc2.weight.copy_(mtp_dict[f"model.layers.{mtp_layer_idx}.mlp.experts." + str(i) + ".down_proj.weight"])
        shared_fc1_weight = torch.cat(
            [mtp_dict[f"model.layers.{mtp_layer_idx}.mlp.shared_experts.gate_proj.weight"],
                mtp_dict[f"model.layers.{mtp_layer_idx}.mlp.shared_experts.up_proj.weight"]])
        mtplayer.mlp.shared_experts.linear_fc1.weight.copy_(shared_fc1_weight)
        mtplayer.mlp.shared_experts.linear_fc2.weight.copy_(mtp_dict[f"model.layers.{mtp_layer_idx}.mlp.shared_experts.down_proj.weight"])
        # output norm
        mgmodel.mtp.layers[0].final_layernorm.weight.copy_(mtp_dict[f"model.layers.{mtp_layer_idx}.shared_head.norm.weight"])
        # NOTE: no need to copy model.layers.{mtp_layer_idx}.shared_head.head.weight

    mgmodel.embedding.word_embeddings.weight.copy_(hfmodel.model.embed_tokens.weight)
    for layer_idx, (mglayer, hflayer) in enumerate(zip(mgmodel.decoder.layers, hfmodel.model.layers)):
        print(layer_idx)
        mglayer.input_layernorm.weight.copy_(hflayer.input_layernorm.weight)

        if args.q_lora_rank is None:
            mglayer.self_attention.linear_q_proj.weight.copy_(hflayer.self_attn.q_proj.weight)
        else:
            mglayer.self_attention.linear_q_down_proj.weight.copy_(hflayer.self_attn.q_a_proj.weight)
            mglayer.self_attention.linear_q_up_proj.weight.copy_(hflayer.self_attn.q_b_proj.weight)
            mglayer.self_attention.linear_q_up_proj.layer_norm_weight.copy_(hflayer.self_attn.q_a_layernorm.weight)

        mglayer.self_attention.linear_kv_down_proj.weight.copy_(hflayer.self_attn.kv_a_proj_with_mqa.weight)
        mglayer.self_attention.linear_kv_up_proj.weight.copy_(hflayer.self_attn.kv_b_proj.weight)
        mglayer.self_attention.linear_kv_up_proj.layer_norm_weight.copy_(hflayer.self_attn.kv_a_layernorm.weight)
        mglayer.self_attention.linear_proj.weight.copy_(hflayer.self_attn.o_proj.weight)
        if not hasattr(mglayer.mlp, 'router'):
            mglayer.mlp.linear_fc1.layer_norm_weight.copy_(hflayer.post_attention_layernorm.weight)
            mglayer.mlp.linear_fc1.weight.copy_(
                torch.cat([hflayer.mlp.gate_proj.weight, hflayer.mlp.up_proj.weight]))
            mglayer.mlp.linear_fc2.weight.copy_(hflayer.mlp.down_proj.weight)
        else:
            mglayer.mlp.router.weight.copy_(hflayer.mlp.gate.weight)
            # NOTE: the e_score_correction_bias in mcore model will be initialized with bfloat16 and \
            # recover to fp32 in the first forward. There is always a diff in the bias between two models (~0.3%)
            safe_copy(hflayer.mlp.gate.e_score_correction_bias, mglayer.mlp.router.expert_bias, skip_dtype_assert=True)
            if args.moe_grouped_gemm == True:
                for i, hf_expert in enumerate(hflayer.mlp.experts):
                    fc1_weight = torch.cat([hf_expert.gate_proj.weight, hf_expert.up_proj.weight])
                    linear_fc1_weighti = getattr(mglayer.mlp.experts.linear_fc1, 'weight' + str(i))
                    linear_fc1_weighti.copy_(fc1_weight)
                    linear_fc2_weighti = getattr(mglayer.mlp.experts.linear_fc2, 'weight' + str(i))
                    linear_fc2_weighti.copy_(hf_expert.down_proj.weight)
            else:
                for i, hf_expert in enumerate(hflayer.mlp.experts):
                    expert = mglayer.mlp.experts.local_experts[i]
                    fc1_weight = torch.cat([hf_expert.gate_proj.weight, hf_expert.up_proj.weight])
                    expert.linear_fc1.weight.copy_(fc1_weight)
                    expert.linear_fc2.weight.copy_(hf_expert.down_proj.weight)
            mglayer.pre_mlp_layernorm.weight.copy_(hflayer.post_attention_layernorm.weight)
            shared_fc1_weight = torch.cat(
                [hflayer.mlp.shared_experts.gate_proj.weight, hflayer.mlp.shared_experts.up_proj.weight])
            mglayer.mlp.shared_experts.linear_fc1.weight.copy_(shared_fc1_weight)
            mglayer.mlp.shared_experts.linear_fc2.weight.copy_(hflayer.mlp.shared_experts.down_proj.weight)

        mgmodel.decoder.final_layernorm.weight.copy_(hfmodel.model.norm.weight)
        if args.untie_embeddings_and_output_weights:
            mgmodel.output_layer.weight.copy_(hfmodel.lm_head.weight)

def contains(key, str_list):
    for s in str_list:
        if s in key:
            return True
    return False

def split_column_parallel(tensor, tp_rank, tp_size):
    seg = tensor.shape[0] // tp_size
    return tensor[seg * tp_rank: seg * (tp_rank + 1)]

def split_row_parallel(tensor, tp_rank, tp_size):
    seg = tensor.shape[1] // tp_size
    return tensor[:, seg * tp_rank: seg * (tp_rank + 1)]

def check_layer(layers_to_copy, k):
    pattern = re.compile(r"decoder.layers.\d+")
    res = pattern.findall(k)
    return res and res[0] in layers_to_copy.keys()

def save_mgmodel(mgmodel, args):
    # tp, etp, ep, pp
    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size

    if args.num_experts is not None:
        args.expert_model_parallel_size = args.target_expert_model_parallel_size
        args.expert_tensor_parallel_size = args.target_expert_tensor_parallel_size

    os.makedirs(args.save, exist_ok=True)
    os.system("cp -rf " + args.load + "/*config.json " + args.save)
    os.system("cp -rf " + args.load + "/config* " + args.save)
    os.system("cp -rf " + args.load + "/tokenizer* " + args.save)
    os.system("cp -rf " + args.load + "/*tok* " + args.save)

    tracker_filepath = os.path.join(args.save, 'latest_checkpointed_iteration.txt')
    with open(tracker_filepath, "w") as f:
        f.write("release")

    full_model = mgmodel.state_dict_for_save_checkpoint()
    for k in list(full_model.keys()):
        if 'extra_state' in k:
            # NOTE: since TE 1.14, fp8 metadata will be saved as tensor. 
            # Always drop these values in the MG ckpt to avoid potential issue.
            # This should work fine because fp8 metadata is not supported by HF ckpt.
            full_model[k] = None
        elif full_model[k] is None:
            full_model.pop(k)

    if args.num_experts is not None:
        if args.moe_grouped_gemm == True:
            pattern = r'weight(\d+)'
        else:
            pattern = r'local_experts\.(\d+)\.'
        assert args.num_experts % args.expert_model_parallel_size == 0
        num_local_experts = args.num_experts // args.expert_model_parallel_size if args.num_experts else 0

    if args.target_decoder_first_pipeline_num_layers is not None:
        assert args.pipeline_model_parallel_size > 1, "decoder_first_pipeline_num_layers is only valid when pp_size > 1"
        remained_layers = args.num_layers - args.target_decoder_first_pipeline_num_layers
        remained_stages = args.pipeline_model_parallel_size - 1
        assert remained_layers % remained_stages == 0
        pp_layers_per_stage = [ args.target_decoder_first_pipeline_num_layers] +([remained_layers // remained_stages] * remained_stages)
    else:
        pp_layers_per_stage = [args.num_layers // args.pipeline_model_parallel_size] * args.pipeline_model_parallel_size

    tp_size = args.tensor_model_parallel_size
    etp_size = args.expert_tensor_parallel_size
    for (tp_rank, etp_rank, ep_rank, pp_rank) in generate_rank_group(
        args.tensor_model_parallel_size,
        args.expert_tensor_parallel_size,
        args.expert_model_parallel_size,
        args.pipeline_model_parallel_size
    ):
        model_split = {}
        layer_offset = sum(pp_layers_per_stage[:pp_rank])
        layers_to_copy = {}
        for layer in range(pp_layers_per_stage[pp_rank]):
            pp_layer_id = layer + layer_offset
            layers_to_copy[f"decoder.layers.{pp_layer_id}"] = layer
        checkpoint_name = get_checkpoint_name(
            args.save, 0, True, 
            args.pipeline_model_parallel_size > 1, 
            tp_rank, 
            pp_rank, 
            args.expert_model_parallel_size > 1, 
            ep_rank
        )
        print(f'save model to {checkpoint_name}')
        has_mtp = pp_rank == args.pipeline_model_parallel_size - 1
        for k, v in full_model.items():
            # NOTE: If k not in current pp_rank, skipping
            if check_layer(layers_to_copy, k):
                layer_pattern = re.compile(r'\d+')
                res = layer_pattern.findall(k)
                k = re.sub(r"decoder.layers.\d+", "decoder.layers." + str(layers_to_copy["decoder.layers." + res[0]]), k)
            elif 'mtp' in k:
                if not has_mtp:
                    continue
            elif not contains(k, ["word_embeddings", "output_layer", "final_layernorm"]):
                continue

            if not isinstance(v, torch.Tensor):
                target_v = v
            elif contains(k, ['linear_q_down_proj', 'linear_kv_down_proj', 'linear_q_proj', 'linear_q_up_proj', 'linear_kv_up_proj', 'linear_q_proj']) and 'norm' not in k:
                target_v = split_column_parallel(v, tp_rank, tp_size)
            elif 'linear_proj' in k:
                target_v = split_row_parallel(v, tp_rank, tp_size)
            elif 'mlp.linear_fc2' in k: # down proj in Dense Layer
                target_v = split_row_parallel(v, tp_rank, tp_size)
            elif 'mlp.linear_fc1' in k and 'norm' not in k: # gate_up proj in Dense Layer
                # Split Gated Column Linear
                seg = args.ffn_hidden_size // args.tensor_model_parallel_size
                viewed = v.view(-1, args.ffn_hidden_size, args.hidden_size)
                target_v = viewed[:, seg * tp_rank: seg * (tp_rank + 1), :].reshape(-1, args.hidden_size)
            elif 'experts' in k and 'shared_experts' not in k:
                # NOTE: If k not in current ep_rank, skipping
                expert_rank = int(re.findall(pattern, k)[0])
                if expert_rank // num_local_experts != ep_rank:
                    continue
                expert_local_rank = expert_rank % num_local_experts
                if args.moe_grouped_gemm == True:
                    k = k.replace(f'weight{expert_rank}', f'weight{expert_local_rank}')
                else:
                    k = k.replace(f'local_experts.{expert_rank}', f'local_experts.{expert_local_rank}')
                if 'linear_fc1' in k:
                    viewed = v.view(-1, args.moe_ffn_hidden_size, args.hidden_size)
                    seg = args.moe_ffn_hidden_size // etp_size
                    target_v = viewed[:, seg * etp_rank: seg * (etp_rank + 1), :].reshape(-1, args.hidden_size)
                elif 'linear_fc2' in k:
                    target_v = split_row_parallel(v, etp_rank, etp_size)
                else:
                    raise NotImplementedError()
            elif 'shared_experts' in k and 'gate' not in k:
                # SharedExperts is from MLP, split by tp_rank
                if 'linear_fc1' in k:
                    viewed = v.view(-1, args.moe_shared_expert_intermediate_size, args.hidden_size)
                    seg = args.moe_shared_expert_intermediate_size // tp_size
                    target_v = viewed[:, seg * tp_rank: seg * (tp_rank + 1), :].reshape(-1, args.hidden_size)
                elif 'linear_fc2' in k:
                    target_v = split_row_parallel(v, tp_rank, tp_size)
                else:
                    raise NotImplementedError()
            elif "word_embeddings" in k or "output_layer" in k:
                target_v = split_column_parallel(v, tp_rank, tp_size)
            elif 'eh_proj' in k:
                target_v = split_column_parallel(v, tp_rank, tp_size)
            else:
                target_v = v

            if "embedding.word_embeddings" in k:
                if pp_rank == 0 or (args.mtp_num_layers > 0 and has_mtp):
                    model_split[k] = target_v
            elif "output_layer" in k or "final_layernorm" in k:
                if pp_rank == args.pipeline_model_parallel_size - 1:
                    model_split[k] = target_v
            else:
                model_split[k] = target_v
        save_state_dict(args, [model_split], checkpoint_name)
    print(f'megatron model is save to {args.save}')

def add_extra_args(parser):
    parser = get_patch_args(parser)
    parser = add_model_args(parser)
    return parser

@torch.inference_mode()
def main():
    initialize_megatron(extra_args_provider=add_extra_args)
    args = get_args()
    generate_rank_group(args.target_tensor_model_parallel_size,
                        args.target_expert_tensor_parallel_size,
                        args.target_expert_model_parallel_size,
                        args.target_pipeline_model_parallel_size)
    assert args.mtp_num_layers in [None, 1], "Currently only support conversion with no more than 1 MTP head."
    if args.convert_checkpoint_from_megatron_to_transformers:
        config = AutoConfig.from_pretrained(args.hf_ckpt_path, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_pretrained(args.hf_ckpt_path, trust_remote_code=True, torch_dtype=config.torch_dtype)
        for p in hf_model.parameters():
            p.fill_(torch.nan)
        mg_model = load_megatron_model(args)
        # NOTE: return a state_dict for saving (to support MTP)
        hf_model = convert_checkpoint_from_megatron_to_transformers(mg_model, hf_model, args)
        save_hfmodel(args, hf_model)
    else:
        config = AutoConfig.from_pretrained(args.load, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=args.load, trust_remote_code=True, torch_dtype=config.torch_dtype)
        mg_model = model_provider()
        for p in mg_model.parameters():
            p.fill_(torch.nan)
        convert_checkpoint_from_transformers_to_megatron(hf_model, mg_model, args)
        del hf_model
        gc.collect()
        save_mgmodel(mg_model, args)

if __name__ == "__main__":

    main()
import os
import shutil
import torch
import json
import logging

from torch import distributed as dist
from safetensors import safe_open
from transformers.modeling_utils import SAFE_WEIGHTS_INDEX_NAME
from huggingface_hub.constants import SAFETENSORS_INDEX_FILE

from megatron.training.checkpointing import (
    save_checkpoint, 
    get_checkpoint_tracker_filename, 
    get_checkpoint_name
)

from general.synchronizer import BaseSynchronizer, ParamType

class HF2MGSynchronizer(BaseSynchronizer):

    def __init__(self, load_dir, model_provider_func=None):
        super().__init__(load_dir, model_provider_func)
        self._single_file = False
        p = os.path.join(self.load_dir, SAFE_WEIGHTS_INDEX_NAME)
        if not os.path.exists(p):
            p = os.path.join(self.load_dir, SAFETENSORS_INDEX_FILE)
        
        if os.path.exists(p):
            with open(p, 'r') as f:
                data = json.load(f)['weight_map']
                self._key_to_file = {k: os.path.join(self.load_dir, v) for k, v in data.items()}
        elif os.path.exists(os.path.join(self.load_dir, 'model.safetensors')):
            self._single_file = True
            self._key_to_file = dict()
        else:
            raise FileNotFoundError()
        if self.debug:
            if not self.dryrun:
                for p in self._mgmodel.parameters():
                    p.data.fill_(torch.nan)
                # NOTE: Fill non-persistent/persistent buffer with NaN
                for b in self._mgmodel.buffers():
                    b.data.fill_(torch.nan)
            self._visit = torch.zeros([self.hf_size], dtype=torch.int, device=self.device)

    def load_tensor(self, dummy_tensor):
        def _get_filename_from_key(key):
            if self._single_file:
                return os.path.join(self.load_dir, 'model.safetensors')
            if key in self._key_to_file:
                return self._key_to_file[key]
            raise KeyError(f'{key} not found in index file')

        if dummy_tensor not in self._hf_params_to_key:
            raise ValueError()
        key = self._hf_params_to_key[dummy_tensor]
        if self.debug:
            self._visit[self._hf_params_key_to_id[key]] = True
        if not self.args.untie_embeddings_and_output_weights and key == 'lm_head.weight':
            key = 'model.embed_tokens.weight'
        file = _get_filename_from_key(key)
        with safe_open(file, framework="pt", device=str(self.device)) as f:
            return f.get_tensor(key)
    
    def _copy_impl(
        self, 
        src_tensor, 
        dst_tensor, 
        param_type: ParamType=ParamType.UNIQUE
    ):
        tp_rank, tp_size = self.tp_rank, self.tp_size
        if param_type in [ParamType.MOE_COLUMN, ParamType.MOE_ROW, ParamType.MOE_GATE_UP]:
            tp_rank, tp_size = self.etp_rank, self.etp_size
        split_mapping = {
            ParamType.UNIQUE: lambda x: self.load_tensor(x),
            ParamType.COLUMN: lambda x: torch.chunk(self.load_tensor(x), tp_size, dim=0)[tp_rank],
            ParamType.ROW: lambda x: torch.chunk(self.load_tensor(x), tp_size, dim=1)[tp_rank],
            # the data of following type is loaded by caller
            ParamType.GATE_UP: lambda x: torch.chunk(x, tp_size, dim=1)[tp_rank].flatten(0, 1),
            ParamType.QKV_W: lambda x: torch.chunk(x, tp_size, dim=0)[tp_rank].flatten(0, 1),
            ParamType.QKV_B: lambda x: torch.chunk(x, tp_size, dim=0)[tp_rank].flatten(),
            ParamType.MOE_COLUMN: lambda x: torch.chunk(self.load_tensor(x), tp_size, dim=0)[tp_rank],
            ParamType.MOE_ROW: lambda x: torch.chunk(self.load_tensor(x), tp_size, dim=1)[tp_rank],
            # the data of following type is loaded by caller
            ParamType.MOE_GATE_UP: lambda x: torch.chunk(x, tp_size, dim=1)[tp_rank].flatten(0, 1),
        }
        if self.dryrun:
            return dst_tensor.data.copy_(dst_tensor.clone())
        dst_tensor.data.copy_(split_mapping[param_type](src_tensor))

    def set_preprocess_state(self):
        '''Set embedding params.'''
        self.copy(
            self._hfmodel.model.embed_tokens.weight, 
            self._mgmodel.embedding.word_embeddings.weight, 
            param_type=ParamType.COLUMN
        )

    def set_postprocess_state(self):
        '''Set output layer & norm params.'''
        self.copy(
            self._hfmodel.model.norm.weight, 
            self._mgmodel.decoder.final_layernorm.weight, 
        )
        if self._mgmodel.share_embeddings_and_output_weights:
            output_layer_weight = self._mgmodel.shared_embedding_or_output_weight() 
        else:
            output_layer_weight = self._mgmodel.output_layer.weight
        self.copy(
            self._hfmodel.lm_head.weight, 
            output_layer_weight, 
            param_type=ParamType.COLUMN
        )

    def set_mla_selfattn_state(self, attn, hf_attn):
        # NOTE: MLA qkv_bias always False
        if self.args.q_lora_rank is None:
            self.copy(hf_attn.q_proj.weight, attn.linear_q_proj.weight, param_type=ParamType.COLUMN)
        else:
            self.copy(hf_attn.q_a_proj.weight, attn.linear_q_down_proj.weight, param_type=ParamType.COLUMN)
            self.copy(hf_attn.q_b_proj.weight, attn.linear_q_up_proj.weight, param_type=ParamType.COLUMN)
            if self.args.qk_layernorm:
                self.copy(
                    hf_attn.q_a_layernorm.weight, 
                    attn.linear_q_up_proj.layer_norm_weight
                )

        self.copy(hf_attn.kv_a_proj_with_mqa.weight, attn.linear_kv_down_proj.weight, param_type=ParamType.COLUMN)
        self.copy(hf_attn.kv_b_proj.weight, attn.linear_kv_up_proj.weight, param_type=ParamType.COLUMN)
        if self.args.qk_layernorm:
            self.copy(
                hf_attn.kv_a_layernorm.weight, 
                attn.linear_kv_up_proj.layer_norm_weight
            )

        self.copy(
            hf_attn.o_proj.weight,
            attn.linear_proj.weight,
            param_type=ParamType.ROW
        )

    def set_selfattn_state(self, attn, hf_attn):
        '''Set self-attention params.'''
        # Reshape loaded weights.
        num_heads = self.args.num_attention_heads
        num_query_groups = (self.args.num_query_groups if self.args.group_query_attention else self.args.num_attention_heads)
        num_querys_per_group = num_heads // num_query_groups
        dim = self.args.kv_channels
        assert num_heads % num_querys_per_group == 0
        # copy qk norm if indeed.
        if self.args.qk_layernorm:
            self.copy(hf_attn.q_norm.weight, attn.q_layernorm.weight)
            self.copy(hf_attn.k_norm.weight, attn.k_layernorm.weight)

        # Copy weights (re-order dimensions for Megatron).
        if self.dryrun:
            attn_proj_weight = attn.linear_qkv.weight
        else:
            attn_proj_weight = torch.cat([
                self.load_tensor(hf_attn.q_proj.weight).reshape((num_query_groups, num_querys_per_group*dim, -1)),
                self.load_tensor(hf_attn.k_proj.weight).reshape((num_query_groups, dim, -1)),
                self.load_tensor(hf_attn.v_proj.weight).reshape((num_query_groups, dim, -1)),
            ], dim=1)
        self.copy(
            attn_proj_weight, 
            attn.linear_qkv.weight,
            param_type=ParamType.QKV_W,
        )
        self.copy(
            hf_attn.o_proj.weight,
            attn.linear_proj.weight,
            param_type=ParamType.ROW
        )

        # Copy bias
        if self.args.add_qkv_bias:
            if self.dryrun:
                attn_proj_bias = attn.linear_qkv.bias
            else:
                attn_proj_bias = torch.cat([
                    self.load_tensor(hf_attn.q_proj.bias).reshape((num_query_groups, num_querys_per_group*dim, -1)),
                    self.load_tensor(hf_attn.k_proj.bias).reshape((num_query_groups, dim, -1)),
                    self.load_tensor(hf_attn.v_proj.bias).reshape((num_query_groups, dim, -1)),
                ], dim=1)
            self.copy(
                attn_proj_bias, 
                attn.linear_qkv.bias,
                param_type=ParamType.QKV_B,
            )

    def set_mlp_state(self, mlp, hf_mlp, expert_id=''):
        '''
        Set MLP params.
        The mlp (mcore MLP) should have attributes `linear_fc1` and `linear_fc2`.
        Currently only Gated Linear is supported.
        '''
        if self.dryrun:
            gate_up_proj_weight = mlp.linear_fc1.weight
        else:
            gate_up_proj_weight = torch.stack([
                self.load_tensor(hf_mlp.gate_proj.weight),
                self.load_tensor(hf_mlp.up_proj.weight)
            ])
        linear_fc1_weight = getattr(mlp.linear_fc1, f'weight{expert_id}')
        linear_fc2_weight = getattr(mlp.linear_fc2, f'weight{expert_id}')
        self.copy(
            gate_up_proj_weight, 
            linear_fc1_weight, 
            param_type=ParamType.GATE_UP if expert_id == '' else ParamType.MOE_GATE_UP
        )
        self.copy(
            hf_mlp.down_proj.weight, 
            linear_fc2_weight, 
            param_type=ParamType.ROW if expert_id == '' else ParamType.MOE_ROW
        )

    def set_sequential_mlp_state(self, experts, hf_experts):
        '''Set MOE MLP params.'''
        experts = experts.local_experts
        for mg_expert_id, hf_expert_id in self._build_expert_parallel_mapping().items():
            self.set_mlp_state(experts[mg_expert_id], hf_experts[hf_expert_id])

    def set_group_mlp_state(self, experts, hf_experts):
        for mg_expert_id, hf_expert_id in self._build_expert_parallel_mapping().items():
            self.set_mlp_state(experts, hf_experts[hf_expert_id], expert_id=mg_expert_id)
            
    def set_moe_layer_state(self, moe, hf_moe):
        # router
        self.copy(hf_moe.gate.weight, moe.router.weight)
        if moe.router.enable_expert_bias:
            self.copy(hf_moe.gate.e_score_correction_bias, moe.router.expert_bias)
        # experts
        if self.args.moe_grouped_gemm:
            # group gemm
            if self.args.moe_use_legacy_grouped_gemm:
                # weight1 and weight2, not impl
                raise NotImplementedError("Currently only TE GroupGEMM is implemented.")
            self.set_group_mlp_state(moe.experts, hf_moe.experts)
        else:
            # sequential
            self.set_sequential_mlp_state(moe.experts, hf_moe.experts)

        # shared experts
        if moe.shared_experts is not None:
            if moe.shared_experts.use_shared_expert_gate:
                self.copy(hf_moe.shared_expert_gate.weight, moe.shared_experts.gate_weight)
            self.set_mlp_state(moe.shared_experts, hf_moe.shared_experts)

    def set_layer_state(self, layer, hf_layer):
        '''Set transformer layer params.'''
        if self.args.multi_latent_attention:
            self.set_mla_selfattn_state(layer.self_attention, hf_layer.self_attn)
            self.copy(hf_layer.input_layernorm.weight, layer.input_layernorm.weight)
        else:
            self.set_selfattn_state(layer.self_attention, hf_layer.self_attn)
            self.copy(hf_layer.input_layernorm.weight, layer.self_attention.linear_qkv.layer_norm_weight)
        
        if hasattr(layer.mlp, 'router'):
            self.set_moe_layer_state(layer.mlp, hf_layer.mlp)
            self.copy(hf_layer.post_attention_layernorm.weight, layer.pre_mlp_layernorm.weight)
        else:
            self.set_mlp_state(layer.mlp, hf_layer.mlp)
            self.copy(hf_layer.post_attention_layernorm.weight, layer.mlp.linear_fc1.layer_norm_weight)

    def check_and_save(self, output_dir):
        if self.debug:
            if not self.dryrun:
                for n, p in self._mgmodel.state_dict().items():
                    if isinstance(p, torch.Tensor) and p.isnan().any():
                        raise SystemError(f'NaN Parameters Detected on key {n}')
            
            from torch.distributed import all_reduce
            all_reduce(self._visit)
            unvisit_param_ids = (self._visit == 0).nonzero()[:, 0].cpu().numpy().tolist()
            unvisit_keys = []
            for param_id in unvisit_param_ids:
                unvisit_keys.append(self._id_to_hf_params_key[param_id])
            if len(unvisit_keys) > 0:
                logging.warning(f"Never visit the following huggingface weights in the conversion: {unvisit_keys}")
                
        self.args.save = output_dir
        if not self.dryrun:
            save_checkpoint(
                getattr(self.args, 'iteration', 1),
                [self._mgmodel], None, None, 0,
                pipeline_rank=self.pp_rank, 
                pipeline_parallel=self.pp_size > 1,
                expert_rank=self.ep_rank, 
                expert_parallel=self.ep_size > 1,
                tensor_rank=self.tp_rank
            )

            dist.barrier()
            if self.rank == 0:
                # NOTE: The `save_checkpoint` API can only save a checkpoint in release=False,
                # reset the metadata. (Otherwise user may find their training starts at step 2)
                tracker_filename = get_checkpoint_tracker_filename(self.args.save)
                with open(tracker_filename, 'w') as f:
                    f.write('release')
                source_dir = get_checkpoint_name(self.args.save, 1, False, return_base_dir=True)
                target_dir = get_checkpoint_name(self.args.save, -1, True, return_base_dir=True)
                shutil.move(source_dir, target_dir)

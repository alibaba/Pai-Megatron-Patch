import torch
import logging
from typing import Dict
from general.h2m_synchronizer import HF2MGSynchronizer as _HF2MGSynchronizer
from general.synchronizer import ParamType

class HF2MGSynchronizer(_HF2MGSynchronizer):
    # TODO: to be refactored to Hybrid Model convertor
    def __init__(self, load_dir, model_provider_func=None):
        super().__init__(load_dir, model_provider_func)
        self.layout = self.get_hybrid_layout()
        

    def get_hybrid_layout(self) -> str:
        assert self.args.hybrid_override_pattern is not None
        return self.args.hybrid_override_pattern


    def sync_params(self, mg_model = None, hf_model = None):
        # assume TE backend
        if self.args.transformer_impl != "transformer_engine":
            raise NotImplementedError("Currently only TE model is implemented.")
        
        if mg_model is None:
            mg_model = self._mgmodel
        if hf_model is None:
            hf_model = self._hfmodel

        if mg_model.pre_process:
            self.set_preprocess_state(mg_model=mg_model, hf_model=hf_model)
        
        if mg_model.post_process:
            self.set_postprocess_state(mg_model=mg_model, hf_model=hf_model, is_mamba=True)

        for mg_layer_id, global_mg_layer_id in self._build_pipeline_parallel_mapping().items():
            hf_layer_id  = global_mg_layer_id // 2
            if self.tp_rank == 0 and self.ep_rank == 0 and self.etp_rank == 0:
                logging.info(f"Converting layer {hf_layer_id}")
            
            layer = mg_model.decoder.layers[mg_layer_id]
            hf_layer = hf_model.model.layers[hf_layer_id]

            if self.layout[global_mg_layer_id] == 'M':
                # Mamba layer
                self.set_mamba_layer_state(layer.mixer, hf_layer.linear_attn)
                self.copy(hf_layer.input_layernorm.weight, layer.norm.weight)
            elif self.layout[global_mg_layer_id] == '-':
                # transformer_layer of MLP
                self.set_moe_layer_state(layer.mlp, hf_layer.mlp)
                self.copy(hf_layer.post_attention_layernorm.weight, layer.pre_mlp_layernorm.weight)
            elif self.layout[global_mg_layer_id] == '*':
                # transformer_layer of Attention
                self.set_gated_selfattn_state(layer.self_attention, hf_layer.self_attn)
                self.copy(hf_layer.input_layernorm.weight, layer.self_attention.linear_qgkv.layer_norm_weight)
            else:
                raise ValueError(f"Unrecognized layer type {self.layout[global_mg_layer_id]} in {self.layout}")

    def set_mamba_layer_state(self, mixer, hf_mixer):
        # copy linear_attn to mamba mixer
        ...
    

    def _build_pipeline_parallel_mapping(self) -> Dict[int, int]:
        remained_num_layers = self.args.num_layers
        remained_stages = self.pp_size
        pp_layers_per_stage = [remained_num_layers // remained_stages] * remained_stages

        pp_mapping = {
            i: v for i, v in enumerate(
                range(
                    sum(pp_layers_per_stage[:self.pp_rank]), 
                    sum(pp_layers_per_stage[:self.pp_rank + 1])
                )
            )
        }
        return pp_mapping

    def set_gated_selfattn_state(self, attn, hf_attn):
        '''Set gated self-attention params.'''
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
            attn_proj_weight = attn.linear_qgkv.weight
        else:
            attn_proj_weight = torch.cat([
                self.load_tensor(hf_attn.q_proj.weight).reshape((num_query_groups, 2*num_querys_per_group*dim, -1)),
                self.load_tensor(hf_attn.k_proj.weight).reshape((num_query_groups, dim, -1)),
                self.load_tensor(hf_attn.v_proj.weight).reshape((num_query_groups, dim, -1)),
            ], dim=1)
        self.copy(
            attn_proj_weight, 
            attn.linear_qgkv.weight,
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
                attn_proj_bias = attn.linear_qgkv.bias
            else:
                attn_proj_bias = torch.cat([
                    self.load_tensor(hf_attn.q_proj.bias).reshape((num_query_groups, num_querys_per_group*dim, -1)),
                    self.load_tensor(hf_attn.k_proj.bias).reshape((num_query_groups, dim, -1)),
                    self.load_tensor(hf_attn.v_proj.bias).reshape((num_query_groups, dim, -1)),
                ], dim=1)
            self.copy(
                attn_proj_bias, 
                attn.linear_qgkv.bias,
                param_type=ParamType.QKV_B,
            )
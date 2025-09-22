import logging
from typing import Dict
from general.h2m_synchronizer import HF2MGSynchronizer as _HF2MGSynchronizer


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
            self.set_postprocess_state(mg_model=mg_model, hf_model=hf_model)

        for mg_layer_id, hf_layer_id in self._build_pipeline_parallel_mapping().items():
            if self.tp_rank == 0 and self.ep_rank == 0 and self.etp_rank == 0:
                logging.info(f"Converting layer {hf_layer_id}")
            
            global_mg_layer_id = hf_layer_id * 2
            layer = mg_model.decoder.layers[mg_layer_id]
            hf_layer = hf_model.model.layers[hf_layer_id]

            if self.layout[global_mg_layer_id] == 'M':
                # Mamba layer
                self.set_mamba_layer_state(layer.mixer, hf_layer.linear_attn)
                self.copy(hf_layer.input_layernorm.weight, layer.norm.weight)
            elif self.layout[global_mg_layer_id] == '-':
                # transformer_layer of MLP
                self.set_mlp_state(layer.mlp, hf_layer.mlp)
                self.copy(hf_layer.post_attention_layernorm.weight, layer.mlp.linear_fc1.layer_norm_weight)
            elif self.layout[global_mg_layer_id] == '*':
                # transformer_layer of Attention
                self.set_selfattn_state(layer.self_attention, hf_layer.self_attn)
                self.copy(hf_layer.input_layernorm.weight, layer.input_layernorm.weight)
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
            i: v // 2 for i, v in enumerate(
                range(
                    sum(pp_layers_per_stage[:self.pp_rank]) * 2, 
                    sum(pp_layers_per_stage[:self.pp_rank + 1]) * 2
                )
            )
        }
        return pp_mapping
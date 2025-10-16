from qwen2_5_vl.h2m_synchronizer import HF2MGSynchronizer as _HF2MGSynchronizer
from general.synchronizer import ParamType

class HF2MGSynchronizer(_HF2MGSynchronizer):

    def set_vision_model_layer_state(self, vision_model, hf_vision_model):
        self.copy(
            hf_vision_model.patch_embed.proj.weight,
            vision_model.patch_embed.proj.weight
        )
        self.copy(
            hf_vision_model.patch_embed.proj.bias,
            vision_model.patch_embed.proj.bias
        )
        self.copy(
            hf_vision_model.pos_embed.weight,
            vision_model.pos_embed.weight
        )

        for layer, hf_layer in zip(
            vision_model.decoder.layers,
            hf_vision_model.blocks
        ):
            self.set_vision_layer_state(layer, hf_layer)
        
        self.copy(hf_vision_model.merger.norm.weight, vision_model.decoder.final_layernorm.weight)
        self.copy(hf_vision_model.merger.norm.bias, vision_model.decoder.final_layernorm.bias)
        self.set_mlp_state(vision_model.projection.encoder, hf_vision_model.merger)

        for norm, hf_merger in zip(
            vision_model.decoder.deepstack_norm_list,
            hf_vision_model.deepstack_merger_list
        ):
            self.copy(hf_merger.norm.weight, norm.weight)
            self.copy(hf_merger.norm.bias, norm.bias)

        for merger, hf_merger in zip(
            vision_model.decoder.deepstack_merger_list,
            hf_vision_model.deepstack_merger_list
        ):
            self.set_mlp_state(merger.encoder, hf_merger)

    def set_group_mlp_state(self, experts, hf_experts):
        assert not experts.config.add_bias_linear
        if not self.dryrun:
            gate_up_proj_weight = self.load_tensor(hf_experts.gate_up_proj).permute(0, 2, 1)
            down_proj_weight = self.load_tensor(hf_experts.down_proj).permute(0, 2, 1)

        for mg_expert_id, hf_expert_id in self._build_expert_parallel_mapping().items():
            if self.dryrun:
                hf_gate_up_weight = getattr(experts.linear_fc1, f'weight{mg_expert_id}')
                hf_down_weight = getattr(experts.linear_fc2, f'weight{mg_expert_id}')
            else:
                hf_gate_up_weight = gate_up_proj_weight[hf_expert_id]
                hf_down_weight = down_proj_weight[hf_expert_id]
            
            hf_gate_up_weight = hf_gate_up_weight.reshape(2, -1, hf_gate_up_weight.shape[-1])

            linear_fc1_weight = getattr(experts.linear_fc1, f'weight{mg_expert_id}')
            linear_fc2_weight = getattr(experts.linear_fc2, f'weight{mg_expert_id}')
            self.copy(
                hf_gate_up_weight, 
                linear_fc1_weight, 
                param_type=ParamType.MOE_GATE_UP
            )
            self.copy(
                hf_down_weight, 
                linear_fc2_weight, 
                param_type=ParamType.MOE_DOWN
            )


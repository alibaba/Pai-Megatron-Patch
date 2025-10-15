import torch
from qwen2_5_vl.m2h_synchronizer import MG2HFSynchronizer as _MG2HFSynchronizer
from general.synchronizer import ParamType

class MG2HFSynchronizer(_MG2HFSynchronizer):

    def set_vision_model_layer_state(self, vision_model, hf_vision_model):
        self.copy(
            vision_model.patch_embed.proj.weight,
            hf_vision_model.patch_embed.proj.weight
        )
        self.copy(
            vision_model.patch_embed.proj.bias,
            hf_vision_model.patch_embed.proj.bias
        )
        self.copy(
            vision_model.pos_embed.weight,
            hf_vision_model.pos_embed.weight
        )

        for layer, hf_layer in zip(
            vision_model.decoder.layers,
            hf_vision_model.blocks
        ):
            self.set_vision_layer_state(layer, hf_layer)
        
        self.copy(vision_model.decoder.final_layernorm.weight, hf_vision_model.merger.norm.weight)
        self.copy(vision_model.decoder.final_layernorm.bias, hf_vision_model.merger.norm.bias)
        self.set_mlp_state(vision_model.projection.encoder, hf_vision_model.merger)

        for norm, hf_merger in zip(
            vision_model.decoder.deepstack_norm_list,
            hf_vision_model.deepstack_merger_list
        ):
            self.copy(norm.weight, hf_merger.norm.weight)
            self.copy(norm.bias, hf_merger.norm.bias)

        for merger, hf_merger in zip(
            vision_model.decoder.deepstack_merger_list,
            hf_vision_model.deepstack_merger_list
        ):
            self.set_mlp_state(merger.encoder, hf_merger)

    def set_group_mlp_state(self, experts, hf_experts):
        gate_up_tensors = {}
        down_tensors = {}
        for mg_expert_id, hf_expert_id in self._build_expert_parallel_mapping().items():
            hidden_size = getattr(experts.linear_fc1, f'weight{mg_expert_id}').shape[-1]
            gate_up_tensors[hf_expert_id] = getattr(experts.linear_fc1, f'weight{mg_expert_id}').reshape(2, -1, hidden_size)
            down_tensors[hf_expert_id] = getattr(experts.linear_fc2, f'weight{mg_expert_id}')

        gate_up_tensors = torch.stack([item[1] for item in sorted(gate_up_tensors.items())])
        down_tensors = torch.stack([item[1] for item in sorted(down_tensors.items())])

        self.copy(
            gate_up_tensors, 
            hf_experts.gate_up_proj,
            param_type=ParamType.MOE_GATE_UP
        )

        self.copy(
            down_tensors,
            hf_experts.down_proj,
            param_type=ParamType.MOE_DOWN
        )
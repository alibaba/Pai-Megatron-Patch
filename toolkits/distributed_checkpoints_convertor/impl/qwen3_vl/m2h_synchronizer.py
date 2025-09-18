from qwen2_5_vl.m2h_synchronizer import MG2HFSynchronizer as _MG2HFSynchronizer

class MG2HFSynchronizer(_MG2HFSynchronizer):

    def set_vision_model_layer_state(self, vision_model, hf_vision_model):
        self.copy(
            vision_model.patch_embed.proj.weight,
            hf_vision_model.patch_embed.proj.weight
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
        self.set_merger_mlp_state(vision_model.projection.encoder, hf_vision_model.merger.mlp)

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
            self.set_merger_mlp_state(merger.encoder, hf_merger.mlp)

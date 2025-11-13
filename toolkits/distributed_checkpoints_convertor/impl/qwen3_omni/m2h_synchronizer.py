import torch
import transformers
from transformers import AutoConfig
from accelerate import init_empty_weights

from qwen3_vl.m2h_synchronizer import MG2HFSynchronizer as _MG2HFSynchronizer
from general.synchronizer import ParamType
from general.m2h_synchronizer import MG2HFSynchronizer as BASE

class MG2HFSynchronizer(_MG2HFSynchronizer):

    def __init__(self, load_dir, model_provider_func=None, skip_hf_initialization=True):
        super().__init__(load_dir, model_provider_func, skip_hf_initialization=skip_hf_initialization)

        config = AutoConfig.from_pretrained(self.load_dir, trust_remote_code=True)
        config.enable_audio_output = False
        with init_empty_weights(include_buffers=True):
            automodel_cls = getattr(transformers, self.args.auto_model)
            self._hfmodel = automodel_cls._from_config(config, torch_dtype=config.torch_dtype)

        self.build_hf_mapping()
        if self.debug:
            self._merge_type: torch.Tensor = torch.zeros([self.hf_size], dtype=torch.int, device=self.device)

    def set_postprocess_state(self, mg_model, hf_model, is_mamba: bool=False):
        '''Set output layer & norm params.'''
        if is_mamba:
            self.copy(
                mg_model.decoder.final_norm.weight, 
                hf_model.model.norm.weight, 
            )
        else:
            self.copy(
                mg_model.decoder.final_layernorm.weight, 
                hf_model.norm.weight, 
            )
        if mg_model.share_embeddings_and_output_weights:
            output_layer_weight = mg_model.shared_embedding_or_output_weight() 
        else:
            output_layer_weight = mg_model.output_layer.weight
        # NOTE: hf_model refers to TextModel of VLM or Model of LLM and does not
        # contain lm_head, visit it by directly calling self._hfmodel
        self.copy(output_layer_weight, self._hfmodel.thinker.lm_head.weight, param_type=ParamType.COLUMN)

    def sync_params(self):
        BASE.sync_params(self, self._mgmodel.language_model, self._hfmodel.thinker.model)
        if self._mgmodel.pre_process:
            self.set_vision_model_layer_state(
                self._mgmodel.vision_model,
                self._hfmodel.thinker.visual
            )
            self.set_audio_encoder_state(
                self._mgmodel.audio_model,
                self._hfmodel.thinker.audio_tower
            )

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
        
        self.copy(vision_model.decoder.final_layernorm.weight, hf_vision_model.merger.ln_q.weight)
        self.copy(vision_model.decoder.final_layernorm.bias, hf_vision_model.merger.ln_q.bias)
        self.set_merger_mlp_state(vision_model.projection.encoder, hf_vision_model.merger.mlp)

        for norm, hf_merger in zip(
            vision_model.decoder.deepstack_norm_list,
            hf_vision_model.merger_list
        ):
            self.copy(norm.weight, hf_merger.ln_q.weight)
            self.copy(norm.bias, hf_merger.ln_q.bias)

        for merger, hf_merger in zip(
            vision_model.decoder.deepstack_merger_list,
            hf_vision_model.merger_list
        ):
            self.set_merger_mlp_state(merger.encoder, hf_merger.mlp)
            
    def set_audio_encoder_state(self, audio_encoder, hf_audio_encoder):
        self.set_weight_and_bias_state(audio_encoder.conv2d1, hf_audio_encoder.conv2d1)
        self.set_weight_and_bias_state(audio_encoder.conv2d2, hf_audio_encoder.conv2d2)
        self.set_weight_and_bias_state(audio_encoder.conv2d3, hf_audio_encoder.conv2d3)

        self.set_weight_and_bias_state(audio_encoder.conv_out, hf_audio_encoder.conv_out)
        self.set_weight_and_bias_state(audio_encoder.proj1, hf_audio_encoder.proj1)
        self.set_weight_and_bias_state(audio_encoder.proj2, hf_audio_encoder.proj2)

        for layer, hf_layer in zip(
            audio_encoder.layers,
            hf_audio_encoder.layers
        ):
            self.set_audio_layer_state(layer, hf_layer)

        self.set_weight_and_bias_state(audio_encoder.ln_post, hf_audio_encoder.ln_post)

    def set_weight_and_bias_state(self, module, hf_module):
        self.copy(module.weight, hf_module.weight)
        if hf_module.bias is not None:
            self.copy(module.bias, hf_module.bias)

    def set_audio_layer_state(self, layer, hf_layer):
        self.set_weight_and_bias_state(layer.self_attn_layer_norm, hf_layer.self_attn_layer_norm)
        self.set_weight_and_bias_state(layer.fc1, hf_layer.fc1)
        self.set_weight_and_bias_state(layer.fc2, hf_layer.fc2)
        self.set_weight_and_bias_state(layer.final_layer_norm, hf_layer.final_layer_norm)

        self.set_weight_and_bias_state(layer.self_attn.k_proj, hf_layer.self_attn.k_proj)
        self.set_weight_and_bias_state(layer.self_attn.v_proj, hf_layer.self_attn.v_proj)
        self.set_weight_and_bias_state(layer.self_attn.q_proj, hf_layer.self_attn.q_proj)
        self.set_weight_and_bias_state(layer.self_attn.out_proj, hf_layer.self_attn.out_proj)

    def set_group_mlp_state(self, experts, hf_experts):
        BASE.set_group_mlp_state(self, experts, hf_experts)
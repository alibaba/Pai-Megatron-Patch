import torch

from transformers import AutoConfig

from general.h2m_synchronizer import HF2MGSynchronizer as _HF2MGSynchronizer
from general.synchronizer import ParamType
from deepseek_v3.patch import add_mtp_layers

class HF2MGSynchronizer(_HF2MGSynchronizer):

    def __init__(self, load_dir, model_provider_func=None):
        super().__init__(load_dir, model_provider_func)
        if self.args.mtp_num_layers is not None:
            config = AutoConfig.from_pretrained(self.load_dir, trust_remote_code=True)
            self.hf_mtp_offset = config.num_hidden_layers
            self._hfmodel = add_mtp_layers(self._hfmodel, config, self.args.mtp_num_layers).to(config.torch_dtype)
            self.build_hf_mapping()
        if self.debug:
            if not self.dryrun:
                for p in self._mgmodel.parameters():
                    p.data.fill_(torch.nan)
                # NOTE: Fill non-persistent/persistent buffer with NaN
                for b in self._mgmodel.buffers():
                    b.data.fill_(torch.nan)
            self._visit = torch.zeros([self.hf_size], dtype=torch.int, device=self.device)

    def sync_params(self):
        super().sync_params()
        if self._mgmodel.mtp_process:
            self.set_preprocess_state()
            for mtp_layer_id in range(self.args.mtp_num_layers):
                hf_mtp_layer_id = self.hf_mtp_offset + mtp_layer_id
                self.copy(
                    self._hfmodel.model.layers[hf_mtp_layer_id].embed_tokens.weight, 
                    self._mgmodel.embedding.word_embeddings.weight,
                    param_type=ParamType.COLUMN
                )
                self.set_mtp_layer_state(
                    self._mgmodel.mtp.layers[mtp_layer_id],
                    self._hfmodel.model.layers[hf_mtp_layer_id]
                )            
                self.copy(
                    self._hfmodel.model.layers[hf_mtp_layer_id].shared_head.head.weight, 
                    self._mgmodel.output_layer.weight, 
                    param_type=ParamType.COLUMN
                )     
 
    def set_mtp_layer_state(self, mtp, hf_mtp):
        self.copy(hf_mtp.enorm.weight, mtp.enorm.weight)
        self.copy(hf_mtp.hnorm.weight, mtp.hnorm.weight)
        self.copy(hf_mtp.eh_proj.weight, mtp.eh_proj.weight, param_type=ParamType.COLUMN)
        self.copy(hf_mtp.shared_head.norm.weight, mtp.final_layernorm.weight)
        self.set_layer_state(mtp.transformer_layer, hf_mtp)
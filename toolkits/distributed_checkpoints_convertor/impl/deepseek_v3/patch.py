from torch import nn
from accelerate import init_empty_weights

class NormedLinear(nn.Module):

    def __init__(self, norm_class, config):
        super().__init__()
        self.norm = norm_class(config.hidden_size, eps=config.rms_norm_eps)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

@init_empty_weights(include_buffers=True)
def add_mtp_layers(hfmodel, config, mtp_num_layers):
    basic_decoder_layer_class = hfmodel.model.layers[-1].__class__
    start_layer_id = config.num_hidden_layers
    
    for mtp_layer_id in range(start_layer_id, start_layer_id + mtp_num_layers):
        hfmodel.model.layers.append(
            basic_decoder_layer_class(config, mtp_layer_id)
        )
        # NOTE: patch some special attributes
        mtp_layer: nn.Module = hfmodel.model.layers[-1]
        mtp_layer.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        norm_class = hfmodel.model.norm.__class__
        mtp_layer.enorm = norm_class(config.hidden_size, eps=config.rms_norm_eps)
        mtp_layer.hnorm = norm_class(config.hidden_size, eps=config.rms_norm_eps)
        mtp_layer.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        mtp_layer.shared_head = NormedLinear(norm_class, config)

    return hfmodel
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock


def get_hidden_output(module, args, output):
    return output[0]


class Qwen2MoeForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        for layer in self.model.layers:
            mlp = MixtralSparseMoeBlock(config)
            mlp.register_forward_hook(get_hidden_output)
            layer.mlp = mlp            


class Qwen2MoeConfig(Qwen2Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # create params used in MixtralSparseMoeBlock
        self.hidden_dim = self.hidden_size
        self.ffn_dim = self.intermediate_size
        self.num_local_experts = kwargs.get('num_local_experts', 0)
        self.top_k = kwargs.get('num_experts_per_tok', 2)


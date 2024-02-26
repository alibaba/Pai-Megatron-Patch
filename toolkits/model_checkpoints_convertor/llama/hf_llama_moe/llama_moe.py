from transformers.models.llama.modeling_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock


def get_hidden_output(module, args, output):
    return output[0]


class LlamaMoeForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        for layer in self.model.layers:
            mlp = MixtralSparseMoeBlock(config)
            mlp.register_forward_hook(get_hidden_output)
            layer.mlp = mlp            


class LlamaMoeConfig(LlamaConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # create params used in MixtralSparseMoeBlock
        self.hidden_dim = self.hidden_size
        self.ffn_dim = self.intermediate_size
        self.num_local_experts = kwargs.get('num_local_experts', 0)
        self.top_k = kwargs.get('num_experts_per_tok', 2)


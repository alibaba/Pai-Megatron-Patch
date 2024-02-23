import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from argparse import Namespace
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.modeling_utils import load_sharded_checkpoint


def build_huggingface_model(model_to_load, compute_dtype, random_init=False):
    config = AutoConfig.from_pretrained(
        model_to_load,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_to_load,
        padding_side="right",
        trust_remote_code=True,
    )
    if random_init:
        model = AutoModelForCausalLM.from_config(
            config=config,
            torch_dtype=compute_dtype,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_to_load,
            config=config,
            torch_dtype=compute_dtype,
            trust_remote_code=True,
            device_map="cpu",
        )
    return config, tokenizer, model.eval()


def build_huggingface_moe_model(
    model_to_load, compute_dtype, num_experts, num_experts_per_tok
):
    config = AutoConfig.from_pretrained(
        model_to_load,
        trust_remote_code=True,
        num_local_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_to_load,
        padding_side="right",
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_config(
        config=config,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    return config, tokenizer, model


def replace_mlp_with_moe(args, model):
    if args.group_query_attention:
        num_key_value_heads = args.num_attention_heads // args.num_query_groups
    else:
        num_key_value_heads = args.num_query_groups

    config = MixtralConfig(
        intermediate_size=args.ffn_hidden_size,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_local_experts=args.num_experts,
        num_key_value_heads=num_key_value_heads,
        rope_theta=args.rotary_base,
        rms_norm_eps=args.norm_epsilon,
        num_experts_per_tok=1,
    )

    def get_hidden_output(module, args, output):
        return output[0]

    for layer in model.model.layers:
        mlp = MixtralSparseMoeBlock(config).to(args.params_dtype)
        mlp.register_forward_hook(get_hidden_output)
        layer.mlp = mlp
    return model


if __name__ == "__main__":
    load_path = "/workdir/01ai/Yi-6B"
    load_path_moe = "/workdir/01ai/hg_moe2_tp1_pp1_ep2"

    args = Namespace(
        ffn_hidden_size=11008,
        hidden_size=4096,
        num_attention_heads=32,
        num_experts=2,
        num_query_groups=4,
        group_query_attention=True,
        rotary_base=500000,
        norm_epsilon=1e-5,
        params_dtype=torch.bfloat16,
    )
    config, tokenizer, model = build_huggingface_model(load_path, args.params_dtype)
    print(f"plain model {model}")
    model = replace_mlp_with_moe(args, model)
    load_sharded_checkpoint(model, load_path_moe)
    print(f"moe model {model}")

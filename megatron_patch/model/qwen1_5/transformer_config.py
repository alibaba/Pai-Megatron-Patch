from dataclasses import dataclass
from megatron.core.transformer import TransformerConfig


@dataclass
class QwenTransformerConfig(TransformerConfig):

    moe_ffn_hidden_size: int = None

    shared_moe_ffn_hidden_size: int = None

    enable_shared_expert: bool = False

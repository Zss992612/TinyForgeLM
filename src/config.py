# src/config.py

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    max_position_embeddings: int

    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    qkv_bias: bool = False
    tie_word_embeddings: bool = True
    
    def __post_init__(self) -> None:
        positive_fields = [
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "max_position_embeddings",
        ]
        for field_name in positive_fields:
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} 必须是正整数")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size 必须能被 num_attention_heads 整除"
            )

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads 必须能被 num_key_value_heads 整除"
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads
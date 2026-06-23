import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig
from .attention import GroupedQueryAttention
from .mlp import SwiGLUMLP
from .rmsnorm import RMSNorm

class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(
            config=config
        )
        self.self_attn = GroupedQueryAttention(config=config)
        self.post_attention_layernorm = RMSNorm(
            config=config
        )
        self.mlp = SwiGLUMLP(config=config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


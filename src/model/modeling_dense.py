import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig
from .embedding import Embedding
from .rmsnorm import RMSNorm
from .block import DecoderBlock

class DenseCausalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(config=config)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(config=config)
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(config=config)
        self.lm_head = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            bias=False,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits
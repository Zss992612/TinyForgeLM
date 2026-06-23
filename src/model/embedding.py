import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import ModelConfig

class Embedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty(
                self.vocab_size,
                self.hidden_size
            )
        )
        self.reset_parameters()
    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.weight,
            mean=0.0,
            std=0.02,
        )
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[input_ids]
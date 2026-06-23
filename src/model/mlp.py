import torch 
import torch.nn as nn
import torch.nn.functional as F
from src.config import ModelConfig

class SwiGLUMLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.intermediate_size,
            bias=False,
        )

        self.up_proj = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.intermediate_size,
            bias=False,
        )

        self.down_proj = nn.Linear(
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            bias=False,
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)

        activated_gate = F.silu(gate)

        hidden_states = activated_gate * up
        output = self.down_proj(hidden_states)

        return output

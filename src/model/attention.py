import torch
import torch.nn as nn

from src.config import ModelConfig
from .rope import build_rope_cache, apply_rotary_pos_emb

def repeat_kv(
        hidden_states: torch.Tensor,
        num_repeats: int
) -> torch.Tensor:
    if num_repeats == 1:
        return hidden_states
    return torch.repeat_interleave(
        hidden_states,
        repeats=num_repeats,
        dim=1,
    )

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = (
            self.num_heads // self.num_kv_heads
        )

        self.rope_theta = config.rope_theta
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.num_heads * self.head_dim,
            bias=config.qkv_bias,
        )

        self.k_proj = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.num_kv_heads * self.head_dim,
            bias=config.qkv_bias,
        )

        self.v_proj = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.num_kv_heads * self.head_dim,
            bias=config.qkv_bias,
        )

        self.o_proj = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            bias=config.qkv_bias,
        )
    
    def forward(
            self,
            hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"输入最后一维应为{self.hidden_size}",
                f"实际输入为{hidden_size}"
            )

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim
        ).transpose(1, 2)
        cos, sin = build_rope_cache(
            seq_len=seq_len,
            head_dim=self.head_dim,
            rope_theta=self.rope_theta,
            device=hidden_states.device,
            dtype=q.dtype,
        )
        q, k = apply_rotary_pos_emb(q, k, cos=cos, sin=sin)
        k = repeat_kv(k, num_repeats=self.num_kv_groups)
        v = repeat_kv(v, num_repeats=self.num_kv_groups)

        attention_score = torch.matmul(
            q,
            k.transpose(-1, -2),
        )
        attention_score = attention_score * self.scale

        causal_mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=hidden_states.device,
                dtype=torch.bool,
            )
        )
        causal_mask = causal_mask.view(
            1,
            1,
            seq_len,
            seq_len,
        )
        attention_score = attention_score.masked_fill(
            ~causal_mask,
            torch.finfo(attention_score.dtype).min,
        )
        attention_weights = torch.softmax(
            attention_score,
            dim=-1,
            dtype=torch.float32,
        ).to(q.dtype)
        attention_output = torch.matmul(attention_weights, v,)
        attention_output = (
            attention_output
            .transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                seq_len,
                hidden_size,
            )
        )
        attention_output = self.o_proj(attention_output)

        return attention_output

    

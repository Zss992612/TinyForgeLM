import torch

def build_rope_cache(
        seq_len: int,
        head_dim: int,
        rope_theta: float,
        device: torch.device,
        dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    if head_dim % 2 != 0:
        raise ValueError("RoPE要求head_dim必须是偶数")
    inv_freq = 1.0 / (
        rope_theta
        **(
            torch.arange(
                0,
                head_dim,
                2,
                device=device,
                dtype=torch.float32,
            )
            / head_dim
        )
    )
    positions = torch.arange(
        seq_len,
        device=device,
        dtype=torch.float32
    )
    freqs = torch.outer(positions, inv_freq)
    angles = torch.cat([freqs, freqs], dim=-1)
    cos = angles.cos()[None, None, :, :].to(dtype = dtype)
    sin = angles.sin()[None, None, :, :].to(dtype = dtype)

    return cos, sin

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_rotated = (
        q * cos + rotate_half(q) * sin
    )

    k_rotated = (
        k * cos + rotate_half(k) * sin
    )
    return q_rotated, k_rotated
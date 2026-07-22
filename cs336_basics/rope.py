import torch
import torch.nn as nn
from einops import einsum


class RoPE(nn.Module):
    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("RoPE requires d_k to be even.")
        inv_freq = theta ** (-torch.arange(0, d_k, 2, device=device) / d_k)
        positions = torch.arange(max_seq_len, device=device)
        angles = einsum(positions, inv_freq, "max_seq_len,d_k->max_seq_len d_k")
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., ::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
        return out

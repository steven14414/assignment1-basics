import torch
import torch.nn as nn
from .rmsnorm import RMSNorm
from .multihead_self_attention import Multihead_self_attention_with_rope
from .swiglu import SwiGLU


class Transformer_block(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = Multihead_self_attention_with_rope(d_model, num_heads, max_seq_len, theta)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x):  # (B,T,D)
        B, T, D = x.shape
        token_positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x

import torch
import torch.nn as nn

from .multihead_self_attention import MultiheadSelfAttention, MultiheadSelfAttentionWithRoPE
from .rmsnorm import RMSNorm
from .swiglu import SiLUFFN, SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        *,
        use_rmsnorm: bool = True,
        post_norm: bool = False,
        use_rope: bool = True,
        ffn_type: str = "swiglu",
    ):
        super().__init__()
        self.use_rmsnorm = use_rmsnorm
        self.post_norm = post_norm
        self.use_rope = use_rope

        if use_rmsnorm:
            self.ln1 = RMSNorm(d_model)
            self.ln2 = RMSNorm(d_model)

        if use_rope:
            self.attn = MultiheadSelfAttentionWithRoPE(d_model, num_heads, max_seq_len, theta)
        else:
            self.attn = MultiheadSelfAttention(d_model, num_heads)

        if ffn_type == "silu":
            self.ffn = SiLUFFN(d_model, d_ff)
        else:
            self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x):  # (B,T,D)
        B, T, D = x.shape
        token_positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)

        if self.use_rope:
            attn_out = self.attn(x if not self.use_rmsnorm or self.post_norm else self.ln1(x), token_positions)
        else:
            attn_in = x if not self.use_rmsnorm or self.post_norm else self.ln1(x)
            attn_out = self.attn(attn_in)

        if self.post_norm:
            x = self.ln1(x + attn_out) if self.use_rmsnorm else x + attn_out
            ffn_in = x
            ffn_out = self.ffn(ffn_in)
            x = self.ln2(x + ffn_out) if self.use_rmsnorm else x + ffn_out
        else:
            x = x + attn_out
            ffn_in = self.ln2(x) if self.use_rmsnorm else x
            x = x + self.ffn(ffn_in)

        return x

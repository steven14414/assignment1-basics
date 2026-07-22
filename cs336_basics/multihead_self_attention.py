import torch
import torch.nn as nn
from einops import rearrange
from .linear import Linear
from .scaled_dot_product_attention import scaled_dot_product_attention
from .rope import Rope


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model // num_heads
        self.qkv_proj = Linear(d_model, d_model * 3)
        self.o_proj = Linear(d_model, d_model)

    def forward(self, x):
        QKV = self.qkv_proj(x)
        Q, K, V = torch.chunk(QKV, chunks=3, dim=-1)
        Q = rearrange(Q, "B T (N H)->B N T H", H=self.d_k, N=self.num_heads)
        K = rearrange(K, "B T (N H)->B N T H", H=self.d_k, N=self.num_heads)
        V = rearrange(V, "B T (N H)->B N T H", H=self.d_v, N=self.num_heads)
        T = Q.shape[-2]
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        out = scaled_dot_product_attention(Q, K, V, mask)
        out = rearrange(out, "B N T H->B T (N H)")
        return self.o_proj(out)


class MultiheadSelfAttentionWithRoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model // num_heads
        self.qkv_proj = Linear(d_model, d_model * 3)
        self.o_proj = Linear(d_model, d_model)
        self.rope = Rope(theta, self.d_k, max_seq_len)

    def forward(self, x, token_positions):
        QKV = self.qkv_proj(x)
        Q, K, V = torch.chunk(QKV, chunks=3, dim=-1)
        Q = rearrange(Q, "B T (N H)->B N T H", H=self.d_k, N=self.num_heads)
        K = rearrange(K, "B T (N H)->B N T H", H=self.d_k, N=self.num_heads)
        V = rearrange(V, "B T (N H)->B N T H", H=self.d_v, N=self.num_heads)
        T = Q.shape[-2]
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        out = scaled_dot_product_attention(Q, K, V, mask)
        out = rearrange(out, "B N T H->B T (N H)")
        return self.o_proj(out)

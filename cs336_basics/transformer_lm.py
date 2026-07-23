import torch.nn as nn

from .embedding import Embedding
from .linear import Linear
from .rmsnorm import RMSNorm
from .transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        *,
        use_rmsnorm: bool = True,
        post_norm: bool = False,
        use_rope: bool = True,
        ffn_type: str = "swiglu",
    ):
        super().__init__()
        self.use_rmsnorm = use_rmsnorm
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                    use_rmsnorm=use_rmsnorm,
                    post_norm=post_norm,
                    use_rope=use_rope,
                    ffn_type=ffn_type,
                )
                for _ in range(num_layers)
            ]
        )
        if use_rmsnorm:
            self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.token_embeddings(x)
        for block in self.layers:
            x = block(x)
        if self.use_rmsnorm:
            x = self.ln_final(x)
        x = self.lm_head(x)
        return x

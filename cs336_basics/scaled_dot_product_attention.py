from math import inf
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from .softmax import softmax


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = K.shape[-1]
    scores = Q @ K.transpose(-2, -1) / (d_k**0.5)
    if mask is not None:
        scores[~mask] = -inf
    return softmax(scores, -1) @ V

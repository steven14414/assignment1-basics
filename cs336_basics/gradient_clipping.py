from collections.abc import Iterable

import torch


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    eps = 1e-6
    grads = [p.grad for p in parameters if p.grad is not None]
    if grads is None:
        return
    total_norm = sum([(grad**2).sum() for grad in grads]) ** 0.5
    clip_coef = max_l2_norm / (total_norm + eps)
    if clip_coef < 1:
        for grad in grads:
            grad.mul_(clip_coef)

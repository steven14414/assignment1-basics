from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss

torch.manual_seed(0)
init_weights = 5 * torch.randn((10, 10))

for lr in [1e1, 1e2, 1e3]:
    weights = torch.nn.Parameter(init_weights.clone())
    opt = SGD([weights], lr=lr)
    losses = []
    for t in range(10):
        opt.zero_grad()
        loss = (weights**2).mean()
        losses.append(loss.item())
        loss.backward()
        opt.step()
    print(f"lr={lr:g}: losses={[f'{x:.4e}' for x in losses]}")
    if losses[-1] < losses[0] * 0.5:
        trend = "decays faster"
    elif losses[-1] < losses[0]:
        trend = "decays"
    elif losses[-1] > losses[0] * 2:
        trend = "diverges"
    else:
        trend = "slow/unstable"
    print(f"  start={losses[0]:.4e}, end={losses[-1]:.4e}, max={max(losses):.4e} -> {trend}")
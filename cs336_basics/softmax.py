import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x - torch.amax(x, dim=dim, keepdim=True)
    x = torch.exp(x)
    x = x / torch.sum(x, dim=dim, keepdim=True)
    return x

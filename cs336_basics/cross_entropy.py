import torch


def cross_entropy(logits, targets):
    logits = logits - torch.amax(logits, dim=-1, keepdim=True)
    log_sum_exp = torch.log(torch.sum(torch.exp(logits), dim=-1))
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    loss = log_sum_exp - target_logits
    return loss.mean()

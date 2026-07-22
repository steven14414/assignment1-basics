import torch


def cross_entropy(logits, targets):
    logits = logits - torch.amax(logits, dim=-1, keepdim=True)
    log_sum_exp = torch.log(torch.sum(torch.exp(logits), dim=-1))
    loss = log_sum_exp - logits[torch.arange(logits.size(0)), targets]
    return loss.mean()

import torch

from .softmax import softmax


@torch.no_grad()
def decoding(
    model,
    token_ids: torch.Tensor,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    end_token_id: int | None = None,
) -> torch.Tensor:
    while token_ids.shape[1] < max_tokens:
        if end_token_id is not None and token_ids[0, -1].item() == end_token_id:
            break

        logits = model(token_ids)  # (batch, seq_len, vocab_size)
        next_logits = logits[:, -1, :]  # (batch, vocab_size)

        if temperature == 0:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            next_logits = next_logits / temperature
            sorted_logits, sorted_indices = torch.sort(next_logits, dim=-1, descending=True)
            sorted_probs = softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, float("-inf"))

            probs = softmax(sorted_logits, dim=-1)
            sampled_sorted_idx = torch.multinomial(probs, num_samples=1)
            next_token = sorted_indices.gather(-1, sampled_sorted_idx)

        token_ids = torch.cat([token_ids, next_token], dim=1)

    return token_ids

from math import cos, pi


def lr_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    if it <= cosine_cycle_iters:
        return min_learning_rate + 1 / 2 * (1 + cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * pi)) * (
            max_learning_rate - min_learning_rate
        )
    else:
        return min_learning_rate

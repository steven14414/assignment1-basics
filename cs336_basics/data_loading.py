import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    start_indices = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    input_indices = start_indices[:, None] + np.arange(0, context_length)[None, :]
    label_indices = input_indices + 1
    inputs = torch.Tensor(dataset[input_indices]).to(device)
    labels = torch.Tensor(dataset[label_indices]).to(device)
    return inputs, labels

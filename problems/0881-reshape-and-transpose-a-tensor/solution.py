import torch

def flatten_then_reshape(x: torch.Tensor, new_shape) -> torch.Tensor:
    return x.reshape(-1)

def transpose_last_two(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(-2, -1)
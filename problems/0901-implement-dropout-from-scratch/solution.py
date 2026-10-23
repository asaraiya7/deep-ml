import torch

def dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    if not training or p == 0:
        return x 
    else:
        mask = torch.rand(x.shape) >= p
        return (x * mask) / (1 - p)

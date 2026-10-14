import torch
import torch.nn as nn


def two_layer_mlp_forward(x, w1, b1, w2, b2):
    """Build a 2-layer MLP, set fixed weights, return scalar output.

    Args:
        x (torch.Tensor): Input of shape (1, 2).
        w1 (torch.Tensor): First Linear weight, shape (2, 2).
        b1 (torch.Tensor): First Linear bias, shape (2,).
        w2 (torch.Tensor): Second Linear weight, shape (1, 2).
        b2 (torch.Tensor): Second Linear bias, shape (1,).

    Returns:
        float: Scalar network output.
    """
    lin1 = nn.Linear(2, 2)
    # Initialize lin1
    with torch.no_grad():
        lin1.weight.copy_(w1)
        lin1.bias.copy_(b1)

    relu = nn.ReLU()

    lin2 = nn.Linear(2, 1)
    # Initialize lin2 
    with torch.no_grad():
        lin2.weight.copy_(w2)
        lin2.bias.copy_(b2)

    net = nn.Sequential(lin1, relu, lin2)

    y = net(x)

    return y.item()

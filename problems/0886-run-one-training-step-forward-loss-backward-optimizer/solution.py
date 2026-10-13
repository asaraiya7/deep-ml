import torch
import torch.nn as nn
import torch.nn.functional as F

def train_one_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, lr: float) -> float:
    # TODO: build an SGD optimizer, run one full forward/loss/backward/step cycle,
    # and return the pre-update loss as a Python float.
    # [Step 1: Initialize optimizer and zero out grads]
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    optim.zero_grad()
    # [Step 2: Initialize loss criterion]
    criterion = nn.MSELoss()
    # [Step 3: Perform forward pass ]
    y_pred = model(x)
    # [Step 4: Compute loss]
    loss = criterion(y_pred, y)
    # [Step 5: Backprop loss]
    loss.backward()
    # [Step 6: Perform single step using gradient descent]
    optim.step()

    return loss.item()

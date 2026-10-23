import torch
import torch.nn as nn


def dropout_demo():
    """Demonstrate Dropout behavior in eval vs train mode.

    Returns:
        tuple: (eval_output, train_nonzero_count)
            eval_output: result of Dropout(ones) in eval mode (identity)
            train_nonzero_count: int count of nonzero elements after Dropout in train mode
    """
    torch.manual_seed(0)
    x= torch.ones(10)
    drop = nn.Dropout(p=0.5)
    model = nn.Sequential(drop)

    # Output in eval mode
    model.eval()
    out_eval = model(x)

    # Output in train mode
    model.train()
    out_train = model(x)

    return (out_eval, torch.count_nonzero(out_train).item())


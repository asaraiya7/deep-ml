import torch

def layer_norm(x, gamma, beta, eps=1e-5):
    mean_x = torch.mean(x, dim=-1, keepdim=True)
    var_x = torch.var(x, dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean_x) / torch.sqrt(var_x + eps) 
    y = gamma * x_norm  + beta

    return y 
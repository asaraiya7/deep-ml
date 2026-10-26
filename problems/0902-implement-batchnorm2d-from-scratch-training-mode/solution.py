import torch

def batchnorm2d(x, gamma, beta, eps=1e-5):
    gamma = gamma.view(1, -1, 1, 1)
    beta = beta.view(1, -1, 1, 1)
    x_mean = torch.mean(x, dim=(0, 2, 3), keepdims=True)
    x_var = torch.var(x, dim=(0, 2, 3), keepdims=True, unbiased=False)
    x_norm = (x - x_mean) / (torch.sqrt(x_var + eps))
    y = gamma * x_norm + beta

    return y 

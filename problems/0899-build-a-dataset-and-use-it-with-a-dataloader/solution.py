import torch
from torch.utils.data import Dataset

class TensorPairDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X : torch.tensor = X
        self.y : torch.tensor = y

        assert self.X.shape[0] == self.y.shape[0], f"Shape mismatch between X: {self.X.shape} and Y: {self.Y.shape}"

    def __len__(self):
       return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])


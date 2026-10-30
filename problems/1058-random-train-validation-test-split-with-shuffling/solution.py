import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    """
    # Random row indices 
    rand_row_indices= np.random.default_rng(seed).permutation(data.shape[0])
    # Split indices
    n = data.shape[0]
    train_end = int(n * train_frac)
    validation_end = train_end + int(n * validation_frac)
    train_indices = rand_row_indices[: train_end]
    val_indices = rand_row_indices[train_end: validation_end]
    test_indices = rand_row_indices[validation_end: ]

    return [data[train_indices, :], data[val_indices, :], data[test_indices, :]]
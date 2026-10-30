import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Perform Layer Normalization.
    """
    X_mean = np.mean(X, axis=-1, keepdims=True)
    X_var = np.var(X, axis=-1, keepdims=True)
    X_norm = (X - X_mean) / (np.sqrt(X_var + epsilon))  
    Y = gamma * X_norm + beta

    return Y  
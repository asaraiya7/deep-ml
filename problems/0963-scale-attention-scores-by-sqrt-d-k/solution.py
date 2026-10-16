import numpy as np


def softmax(X):
    X_norm = X - np.max(X, axis=1, keepdims=True)
    exp_X = np.exp(X_norm)
    return exp_X / (np.sum(exp_X, axis=1, keepdims=True))


def scaled_attention_weights(Q: np.ndarray, K: np.ndarray) -> list:
    """
    Compute scaled dot-product attention weights.

    Args:
        Q: (n_q, d_k) query matrix
        K: (n_k, d_k) key matrix

    Returns:
        Attention weights of shape (n_q, n_k) as a nested list,
        each entry rounded to 4 decimal places.
    """
    scores = Q @ K.T
    sqrt_d_k = np.sqrt(K.shape[-1])
    norm_scores = scores / (sqrt_d_k)
    soft_scores = softmax(norm_scores)

    return np.round(soft_scores, 4).tolist()

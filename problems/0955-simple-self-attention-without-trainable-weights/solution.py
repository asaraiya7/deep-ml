import numpy as np

def softmax(X : np.ndarray):
    X_norm = X - np.max(X, axis=1, keepdims=True)
    exp_X = np.exp(X_norm)
    return exp_X / np.sum(exp_X, axis=1, keepdims=True)


def simple_self_attention(X: list[list[float]]) -> list[list[float]]:
    """
    Compute context vectors using simple self-attention (no trainable weights).

    Args:
        X: Input embeddings of shape (T, d) as a list of lists.

    Returns:
        Context vectors of shape (T, d) as a list of lists.
    """
    # Context vec: softmax(np.dot(X, X^T)) @ X
    X_np = np.array(X)
    scores = np.dot(X_np, X_np.T)
    soft_scores = softmax(scores)
    context_vec = soft_scores @ X_np 
    return context_vec.tolist()
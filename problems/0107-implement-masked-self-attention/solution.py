import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def softmax(X: np.ndarray):
    X_norm = X - np.max(X, axis=1, keepdims=True)
    exp_X = np.exp(X_norm)
    return exp_X / (np.sum(exp_X, axis=1, keepdims=True))


def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Compute masked self-attention.
    """
    d_k =  Q.shape[1]
    raw_scores = Q @ K.T
    scaled_scores = raw_scores / np.sqrt(d_k)
    masked_attn_weights = softmax(scaled_scores + mask)
    masked_attention = masked_attn_weights @ V 

    return masked_attention
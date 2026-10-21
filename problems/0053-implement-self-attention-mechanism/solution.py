import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def softmax(Z):
    Z_norm = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_norm)
    return exp_Z / (np.sum(exp_Z, axis=1, keepdims=True))


def self_attention(Q, K, V):
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
    
    Returns:
        Attention output of shape (seq_len, d_v)
    """
    d_k = Q.shape[1]
    attn_scores = Q @ K.T / np.sqrt(d_k)
    attn_weights = softmax(attn_scores)
    attn = attn_weights @ V

    return attn

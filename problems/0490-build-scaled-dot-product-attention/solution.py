import numpy as np

def softmax(Z):
    Z_norm = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_norm)
    return exp_Z / (np.sum(exp_Z, axis=1, keepdims=True))


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None) -> tuple:
    """
    Compute Scaled Dot-Product Attention.

    Args:
        Q: Query matrix of shape (seq_len_q, d_k)
        K: Key matrix of shape (seq_len_k, d_k)
        V: Value matrix of shape (seq_len_k, d_v)
        mask: Optional binary mask of shape (seq_len_q, seq_len_k)

    Returns:
        Tuple of (output, attention_weights)
    """
    d_k = Q.shape[1]
    attn_scores = (Q @ K.T) / np.sqrt(d_k)
    if mask is not None:
		# Specifically use negative inf as +ve inf leads to overflow / numerical issues 
		# We cannot just multipley by 0 as softmax(0) is not 0 
        attn_scores = np.where(mask == 0, -np.inf, attn_scores)
    attn_weights = softmax(attn_scores)
    attn = attn_weights @ V 

    return (attn, attn_weights)
import numpy as np

def softmax(Z: np.ndarray):
    Z_norm = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_norm)

    return exp_Z / (np.sum(exp_Z, axis=1, keepdims=True))

def sigmoid(Z: np.ndarray):
    neg_exp_Z = np.exp(-Z)

    return 1 / (1 + neg_exp_Z)


def gated_attention(
    X: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    W_g: np.ndarray
) -> np.ndarray:
    """
    Compute Gated Attention output.
    
    Args:
        X: Input tensor of shape (seq_len, d_model)
        W_q: Query projection of shape (d_model, d_k)
        W_k: Key projection of shape (d_model, d_k)
        W_v: Value projection of shape (d_model, d_v)
        W_g: Gate projection of shape (d_model, d_v)
    
    Returns:
        Gated attention output of shape (seq_len, d_v), rounded to 4 decimal places
    
    Hint: First compute standard scaled dot-product attention, then apply
    a sigmoid gate to modulate the output.
    """
    # Step 1: Get query, key and value matrices
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)

    # Step 2: Compute standard SDPA 
    d_k = Q.shape[1]
    attn_scores = (Q @ K.T) / np.sqrt(d_k)
    attn_weights = softmax(attn_scores)
    attn = attn_weights @ V 

    # Apply gating function 
    G = sigmoid(np.dot(X, W_g))
    Y_gated = G * attn 

    return Y_gated

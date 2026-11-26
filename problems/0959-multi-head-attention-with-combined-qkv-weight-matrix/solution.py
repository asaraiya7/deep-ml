import numpy as np 

def softmax(X : np.ndarray):
    X_norm = X - np.max(X, axis=-1, keepdims=True)
    exp_X = np.exp(X_norm)
    return exp_X / np.sum(exp_X, axis=-1, keepdims=True)

def mha_combined_qkv(x, W_qkv, W_out, num_heads):
    """
    Multi-head causal self-attention using a single combined QKV weight matrix.

    Args:
        x: array of shape (batch, num_tokens, d_in)
        W_qkv: array of shape (d_in, 3 * d_out)
        W_out: array of shape (d_out, d_out)
        num_heads: int, number of attention heads (d_out % num_heads == 0)

    Returns:
        Nested list of shape (batch, num_tokens, d_out).
    """
    B, num_toks, _ = x.shape
    d_out = W_out.shape[0]
    head_dim = d_out // num_heads
    # (1) Compute projection directly (w/o using 3 separate np.dot() functions)
    # [B, num_toks, 3 * dout] = [B, num_toks, din] x [din, 3 * dout]
    Q_K_V = x @ W_qkv 
    # (2) Splits last dimension into 3 different parts to get Q, K, V separately 
    # Each is [B, num_toks, dout]
    Q, K, V = np.split(Q_K_V, 3, axis=-1)
    Q, K, V = [t.reshape(B, num_toks, num_heads, head_dim).transpose((0, 2, 1, 3)) for t in (Q, K, V)]
    # (3) Compute per-head scaled dot product attention 
    mask = np.triu(np.full((num_toks, num_toks), -np.inf), k=1)
    raw_scores = Q @ np.transpose(K, (0, 1, 3, 2))
    scaled_scores = raw_scores / np.sqrt(head_dim)
    attn_weights = softmax(scaled_scores + mask)
    attn = attn_weights @ V 
    # (4) Concatenate to remove the head dimension 
    attn_all_heads = attn.transpose(0, 2, 1, 3).reshape(B, num_toks, d_out)    
    # (5) Apply output projection 
    attn = attn_all_heads @ W_out 

    return attn.tolist()
import numpy as np

def attention_dropout(attn_weights, values, dropout_rate, mask):
    """Apply dropout mask to attention weights and compute context vectors."""
    mask = np.array(mask)
    attn_weights = np.array(attn_weights)
    values = np.array(values)

    # Prepare mask values according to dropout rate 
    mask = mask * (1 / (1 - dropout_rate))
    # Apply mask to attention weigths 
    attn_weights_masked = mask * attn_weights
    # Compute context vectors 
    context_vec = attn_weights_masked @ values 

    return context_vec.tolist()
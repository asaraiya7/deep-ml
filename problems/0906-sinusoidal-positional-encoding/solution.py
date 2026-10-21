import math
import torch

def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
        
    # Final output folding matrix 
    sin_pos_enc_table = torch.zeros((seq_len, d_model), dtype=torch.float32)
    # Create position vec
    position = torch.arange(0, seq_len, 1)
    # Create div term     
    cols = torch.arange(0, d_model // 2, 1)
    log_term = -torch.log(torch.tensor(10000.0))
    div_term = torch.exp(log_term * 2 * cols / d_model)    
    # Calculate angles 
    angles = torch.outer(position, div_term)
    # Calculate positional embeddings 
    sin_term = torch.sin(angles)
    cos_term = torch.cos(angles)

    # Put constructed positional embedding into matrix 
    # sin: even columns, cos: odd coumns 
    sin_pos_enc_table[:, ::2] = sin_term
    sin_pos_enc_table[:, 1::2] = cos_term

    return sin_pos_enc_table
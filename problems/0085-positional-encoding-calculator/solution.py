import numpy as np

def pos_encoding(position: int, d_model: int):
    # Your code here
    pos_encoding = np.zeros((position, d_model))
    # (Step 1): Setup position vector 
    pos_vec = np.arange(0, position, 1)
    # (Step 2:) Setup divisor 
    cols = np.arange(0, np.ceil(d_model // 2), 1)
    div_term = np.exp(-np.log(10000)* 2.0 * cols / d_model)
    # Step 3: Calculate angles 
    angles = np.outer(pos_vec, div_term)
    # Step 4: Build position encoding matrix  
    pos_encoding[:, ::2] = np.sin(angles)
    # Keep only as many cos terms as needed 
    pos_encoding[:, 1::2] = np.cos(angles)[:, :d_model // 2]

    pos_encoding = np.float16(pos_encoding)
    return pos_encoding
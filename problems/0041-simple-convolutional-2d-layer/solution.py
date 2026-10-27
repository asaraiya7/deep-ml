import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    H_in, W_in  = input_matrix.shape
    K_h, K_w = kernel.shape

    # Numpy function to pad matrix 
    input_padded = np.pad(input_matrix, pad_width=padding, mode='constant', constant_values=0)

    # Compute dims of output matrix
    H_out = (H_in + 2 * padding - K_h) // stride + 1 
    W_out = (W_in + 2 * padding - K_w) // stride + 1 
    output_matrix = np.zeros((H_out, W_out), dtype=np.float32)

    # All indices for output matrix 
    indices = list(np.ndindex(output_matrix.shape))

    for r, c in indices:
		# r, c control output matrix movement 
		# r_s, c_s control kernel movement 
        r_s, c_s  = r * stride, c * stride
        output_matrix[r, c] = np.sum((input_padded[r_s : r_s + K_h, c_s : c_s + K_w] * kernel))
    
    return output_matrix
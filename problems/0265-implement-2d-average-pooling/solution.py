import numpy as np 


def avg_pool_2d(input_matrix: list[list[float]], pool_size: int) -> list[list[float]]:
    """
    Perform 2D average pooling on the input matrix.

    Args:
        input_matrix: 2D input array of shape (H, W)
        pool_size: Size of the square pooling window
        
    Returns:
        2D array after average pooling of shape (H//pool_size, W//pool_size)
    """
    input_np = np.array(input_matrix)
    H_in, W_in = input_np.shape
    H_out, W_out = H_in // pool_size, W_in // pool_size
    avg_pool_2d_out = np.zeros((H_out, W_out), dtype=float)
    rows, cols = np.indices((pool_size, pool_size))

    for i in range(0, H_out):
        for j in range(0, W_out):
            wind = input_np[rows + i* pool_size, cols + j* pool_size]
            avg_pool_2d_out[i, j] = np.mean(wind)

    return avg_pool_2d_out







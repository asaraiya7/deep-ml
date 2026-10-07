import numpy as np

def relu(x):
	return np.maximum(0, x)

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	# Residual block: y = F(x) + x 
	# Step 1: Compute residual F(x)
	a1 = relu(w1 @ x)
	z2 = w2 @ a1
	z2+= x
	# Add original value
	y = relu(z2)
	return y 
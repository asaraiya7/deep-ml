import numpy as np
def precision(y_true, y_pred):
	tp = np.sum((y_true == 1) & (y_pred == 1))
	fp = np.sum((y_true == 0) & (y_pred == 1))

	if tp + fp == 0:
		return 0 

	precision_val = tp / (tp + fp)
	
	return precision_val


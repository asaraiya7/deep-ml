import numpy as np

def accuracy_score(y_true, y_pred):
	tp = np.sum((y_true == 1) & (y_pred == 1))
	tn = np.sum((y_true == 0) & (y_pred == 0))
	fp = np.sum((y_true == 0) & (y_pred == 1))
	fn = np.sum((y_true == 1) & (y_pred == 0))

	accuracy = (tp + tn) / (tp + tn + fp + fn)

	return accuracy
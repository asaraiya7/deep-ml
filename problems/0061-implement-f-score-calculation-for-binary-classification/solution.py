import numpy as np

def f_score(y_true, y_pred, beta):
	"""
	Calculate F-Score for a binary classification task.

	:param y_true: Numpy array of true labels
	:param y_pred: Numpy array of predicted labels
	:param beta: The weight of precision in the harmonic mean
	:return: F-Score rounded to three decimal places
	"""
	# Precision = TP / TP + FP 
	# Recall = TP / TP + FN 
	# F1-score = 2 PR / (P + R)
	tp = np.sum((y_true == 1) & (y_pred == 1))
	fp = np.sum((y_true == 0) & (y_pred == 1))
	fn = np.sum((y_true == 1) & (y_pred == 0))

	precision = tp / (tp + fp)
	recall = tp / (tp + fn)

	if (precision + recall) == 0:
		return 0 

	beta_sq = beta ** 2
	f_beta = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)

	return round(f_beta, 3)

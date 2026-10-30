import numpy as np 
from collections import Counter

def confusion_matrix(data):
    # Implement the function here
    data = np.array(data)
    gt = data[:, 0]
    pred = data[:, 1]
    tp = np.sum((gt == 1) & (pred == 1))
    tn = np.sum((gt == 0) & (pred == 0))
    fp = np.sum((gt == 0) & (pred == 1))
    fn = np.sum((gt == 1) & (pred == 0))

    return [[int(tp), int(fn)], [int(fp), int(tn)]]
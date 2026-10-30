import numpy as np 

def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
    actual_np = np.array(actual)
    pred_np = np.array(predicted)

    assert actual_np.shape[0] == pred_np.shape[0], f"Shapes mismatch actual {actual_np.shape[0]}, pred {pred_np.shape[0]}"

    # Compute metrics primitives 
    tp = np.sum((actual_np == 1) & (pred_np == 1))
    fp = np.sum((actual_np == 0) & (pred_np == 1))
    fn = np.sum((actual_np == 1) & (pred_np == 0))
    tn = np.sum((actual_np == 0) & (pred_np == 0))

    # Compute specific metrics 
    accuracy = float((tp + tn) / (tp + tn + fp + fn))
    confusion_matrix = np.array(([tp, fn], [fp, tn]))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = float((2 * precision * recall) / (precision + recall))
    specificity = float(tn / (tn + fp))
    negativePredictive = float(tn / (tn + fn))

    # Implement your code here
    return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)
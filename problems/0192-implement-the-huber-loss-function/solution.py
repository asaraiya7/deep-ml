import numpy as np 

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute the Huber Loss between true and predicted values.

    Args:
        y_true (float | list[float]): Ground truth values
        y_pred (float | list[float]): Predicted values
        delta (float): Transition threshold between MSE and MAE behavior

    Returns:
        float: Average Huber loss
    """
    y_true_np, y_pred_np  = np.array(y_true).reshape(-1, 1), np.array(y_pred).reshape(-1, 1)
    abs_diff = np.abs(y_pred_np - y_true_np)

    # Find indices where we need to compute different loss functions 
    # based on abs diff 
    mse_mask, mae_mask = abs_diff <= delta, abs_diff > delta 

    # Iniitalize loss 
    huber_loss_np = np.zeros((y_true_np.shape[0], 1))       

    # Vector calc. for MSE loss 
    huber_loss_np[mse_mask] = 0.5 * (y_true_np[mse_mask] - y_pred_np[mse_mask]) ** 2
    # Vector calc. for MAE loss 
    huber_loss_np[mae_mask] = delta * (abs_diff[mae_mask] -  (0.5 * delta))

    return float(np.mean(huber_loss_np))

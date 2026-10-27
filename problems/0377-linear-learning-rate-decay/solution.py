def linear_lr_decay(initial_lr: float, end_lr: float, num_steps: int) -> list:
    """
    Generate a linear learning rate decay schedule.
    
    Args:
        initial_lr: Starting learning rate
        end_lr: Final learning rate
        num_steps: Total number of training steps
    
    Returns:
        List of learning rates for each step
    """
    # Handling edge-cases
    if num_steps == 0:
        return []
    elif num_steps == 1:
        return [initial_lr]
    
    decay_rate = (initial_lr - end_lr) / (num_steps - 1)

    lr = initial_lr
    lr_schedule = []
    for i in range(0, num_steps):
        lr_schedule.append(round(lr, 4))
        lr-= decay_rate

    return lr_schedule
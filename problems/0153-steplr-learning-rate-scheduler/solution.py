class StepLRScheduler:
    def __init__(self, initial_lr, step_size, gamma):
        # Initialize initial_lr, step_size, and gamma
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma 

    def get_lr(self, epoch):
        steps = epoch // self.step_size
        decay_factor = self.gamma ** steps
        lr = self.initial_lr * decay_factor
        return round(lr, 4)
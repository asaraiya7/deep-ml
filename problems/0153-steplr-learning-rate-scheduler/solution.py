class StepLRScheduler:
    def __init__(self, initial_lr, step_size, gamma):
        # Initialize initial_lr, step_size, and gamma
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma 

    def get_lr(self, epoch):
        # Calculate and return the learning rate for the given epoch
        step_num = epoch // self.step_size 
        if step_num == 0:
            return self.initial_lr
        lr = self.initial_lr * (self.gamma ** step_num)
        return round(lr, 4)
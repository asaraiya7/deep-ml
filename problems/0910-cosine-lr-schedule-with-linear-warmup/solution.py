import math

def cosine_with_warmup(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    if step >= total_steps:
        return 0.0
    # TODO: piecewise linear warmup, then half-cosine decay                
    if step <= warmup_steps:
        return (step / warmup_steps) * base_lr
    elif step > warmup_steps:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return base_lr * cosine_decay
    
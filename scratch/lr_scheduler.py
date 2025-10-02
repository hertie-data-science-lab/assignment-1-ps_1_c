import math
import numpy as np

def cosine_annealing(initial_lr, epoch, total_epochs, min_lr=0.0):
    """
    Cosine annealing learning rate scheduler.

    Args:
        initial_lr (float): Initial learning rate.
        epoch (int): Current epoch number.
        total_epochs (int): Total number of epochs.
        min_lr (float): Minimum learning rate to reach (default: 0.0).
        
    Returns:
        float: Adjusted learning rate for the current epoch.
    """
    if epoch > total_epochs:
        epoch = total_epochs
    
    lr = min_lr + 0.5 * (initial_lr - min_lr) * (
        1 + math.cos(math.pi * epoch / total_epochs)
    )
<<<<<<< HEAD
    return lr
=======
    return lr
>>>>>>> 1244ec8c9eb572a2f3fddf0571a6a0206945e6c2

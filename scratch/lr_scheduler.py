import numpy as np

def cosine_annealing(initial_lr: float, epoch: int, total_epochs: int, min_lr: float = 0.0) -> float:
    """
    Cosine annealing learning rate scheduler.

    Effect:
        - Learning rate starts high, allowing fast progress.
        - Gradually decreases.
        - Smooth cosine decay avoids abrupt changes that can destabilize training.
        - Often improves convergence and final accuracy compared to constant learning rates.
    
    Args:
        initial_lr (float): Initial learning rate.
        epoch (int): Current epoch number.
        total_epochs (int): Total number of epochs.
        min_lr (float): Minimum learning rate to reach.
        
    Returns:
        float: Adjusted learning rate for the current epoch.
    """
    cos_inner = np.pi * epoch / total_epochs
    lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(cos_inner))
    return lr
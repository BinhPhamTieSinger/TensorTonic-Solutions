import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    # Write code here
    p, y = np.array(p), np.array(y)
    loss = -np.mean(
        y * (1 - p) ** gamma * np.log(p) +
        (1 - y) * (p ** gamma) * np.log(1 - p)
    )
    return loss
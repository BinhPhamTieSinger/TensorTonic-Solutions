import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    # Write code here
    x = np.array(x)
    y = np.zeros((x.shape))
    y[x > 0] = x[x > 0]
    y[x < 0] = x[x < 0]*alpha
    return y
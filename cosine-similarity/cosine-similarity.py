import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    a, b = np.array(a), np.array(b)
    c, d = np.linalg.norm(a), np.linalg.norm(b)
    if (c == 0) or (d == 0):
        return 0
    return (a @ b)/(c * d)
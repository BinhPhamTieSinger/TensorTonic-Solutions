import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """
    # Write code here
    y, split_mask = np.array(y), np.array(split_mask)
    y_left, y_right = y[split_mask], y[split_mask == False]
    entropy_parent = _entropy(y)
    entropy_left = _entropy(y_left)
    entropy_right = _entropy(y_right)
    nL, nR, N = y_left.size, y_right.size, y.size
    return entropy_parent - (nL/N * entropy_left + nR/N * entropy_right)

import numpy as np

def gini_single(t):
    values, counts = np.unique(t, return_counts = True)
    sum = counts.sum(); probs = counts/sum
    return 1 - np.sum(probs**2)

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    # Write code here
    y_left, y_right = np.array(y_left), np.array(y_right)
    gini_left = gini_single(y_left)
    gini_right = gini_single(y_right)
    y = np.append(y_left, y_right)
    N, NL, NR = y.size, y_left.size, y_right.size
    if N == 0:
        return 0.0
    return NL/N*gini_left + NR/N*gini_right
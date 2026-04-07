import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    X_train, X_test = np.array(X_train), np.array(X_test)

    if X_test.size == 0:
        return np.empty((0, k), dtype=int)
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    neighbors = []
    
    for test_point in X_test:
        distances = np.sum((X_train - test_point) ** 2, axis=1)
        sorted_idx = np.argsort(distances)
        k_indices = sorted_idx[:k]
        
        if len(k_indices) < k:
            pad_size = k - len(k_indices)
            k_indices = np.concatenate([k_indices, [-1]*pad_size])
        
        neighbors.append(k_indices)

    return np.array(neighbors)
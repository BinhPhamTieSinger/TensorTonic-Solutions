import numpy as np

def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    recommended, relevant = np.array(recommended), np.array(relevant)
    intersection_top_k = np.intersect1d(recommended[:k], relevant)
    precision = intersection_top_k.size / k
    recall = intersection_top_k.size / relevant.size
    return [precision, recall]
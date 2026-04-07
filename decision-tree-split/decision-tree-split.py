import numpy as np

def gini_split(y):
    classes, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    gini = 1 - np.dot(counts/total, counts/total)
    return gini

def decision_tree_split(X, y):
    X, y = np.array(X), np.array(y)
    gini_parent = gini_split(y)
    
    max_gain = 0
    index_gain = 0
    threshold_gain = 0

    for i in range(X.shape[1]):
        feature = X[:, i]
        uniq = np.unique(feature)
        if len(uniq) == 1:
            continue
        thresholds = (uniq[:-1] + uniq[1:]) / 2
        
        for threshold in thresholds:
            left_mask = feature <= threshold
            right_mask = feature > threshold
            left_y = y[left_mask]
            right_y = y[right_mask]
            
            gini_left = gini_split(left_y)
            gini_right = gini_split(right_y)
            weighted_gini = (left_y.size / y.size) * gini_left + (right_y.size / y.size) * gini_right
            information_gain = gini_parent - weighted_gini
            
            if information_gain > max_gain:
                max_gain = information_gain
                index_gain = i
                threshold_gain = threshold

    return index_gain, threshold_gain
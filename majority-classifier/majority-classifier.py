import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Write code here
    y_train, X_test = np.array(y_train), np.array(X_test)
    values, counts = np.unique(y_train, return_counts = True)
    mode_value = values[np.argmax(counts)]
    if X_test.size == 0:
        return np.array([])
    return np.full((X_test.size), mode_value)
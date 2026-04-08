import numpy as np

def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    # Write code here
    X, y = np.array(X), np.array(y)
    regularized_matrix = X.T @ X + lam * np.eye(X.shape[1])
    inverse_reg_mat = np.linalg.inv(regularized_matrix)
    weight = inverse_reg_mat @ X.T @ y
    return weight
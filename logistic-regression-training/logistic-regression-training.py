import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def _binary_cross_entropy(N, y, y_pred):
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    ### X is R 4*1 and y is R 1*4
    # Write code here
    N = y.size
    ### Set up initial weights and biases
    w = np.zeros(X.shape[1])
    b = 0.0
    ### Start the iterations:
    for step in range(steps):
        ### Feed forwarding
        z = X @ w + b
        p = _sigmoid(z)
        ### Loss calculation
        loss = _binary_cross_entropy(N, y, p)
        ### Derivative calculation
        dz = p - y
        dw = (X.T @ dz) / N
        db = np.mean(dz)
        ### Gradient Descent
        w -= lr * dw
        b -= lr * db
        print(f"step {step}, loss {loss:.4f}")
    return w, b    
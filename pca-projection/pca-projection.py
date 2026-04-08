import numpy as np

def power_iteration(A, num_iters=1000, tol=1e-8):
    n = A.shape[0]
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)

    for _ in range(num_iters):
        v_next = A @ v
        v_next_norm = np.linalg.norm(v_next)
        if v_next_norm == 0:
            break
        v_next = v_next / v_next_norm
        if np.linalg.norm(v - v_next) < tol:
            break
        v = v_next

    eigenvalue = v.T @ A @ v
    return eigenvalue, v


def top_k_eigenvectors(A, k, num_iters=1000):
    A_work = A.copy().astype(float)
    n = A.shape[0]

    eigenvalues = []
    eigenvectors = []

    for _ in range(k):
        val, vec = power_iteration(A_work, num_iters)
        eigenvalues.append(val)
        eigenvectors.append(vec)
        A_work = A_work - val * np.outer(vec, vec)

    return np.array(eigenvalues), np.column_stack(eigenvectors)

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    X = np.array(X)
    n = X.shape[0]
    X_mean = X.mean(axis=0, keepdims=True)
    X_centered = X - X_mean
    C = 1/(n-1) * X_centered.T @ X_centered
    
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    topk_vals = eigvals[idx[:k]]
    topk_vecs = eigvecs[:, idx[:k]]
    print(topk_vecs)

    # topk_vals, topk_vecs = top_k_eigenvectors(C, k)
    # print(topk_vecs)

    return X_centered @ topk_vecs
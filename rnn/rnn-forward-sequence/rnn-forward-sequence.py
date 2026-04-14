import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    # YOUR CODE HERE
    batches, T, input_dim = X.shape
    hidden_states = []
    h_final = []

    for i, batch in enumerate(X):
        h_prev = h_0[i]
        h_batch = []

        for t in range(T):
            x_t = batch[t]
            h_t = np.tanh(x_t @ W_xh.T + h_prev @ W_hh.T + b_h)
            h_batch.append(h_t)
            h_prev = h_t

        hidden_states.append(h_batch)
        h_final.append(h_t)

    return np.array(hidden_states), np.array(h_final)
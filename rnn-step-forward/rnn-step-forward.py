import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    # Write code here
    ### h_t = tanh(Wx*x_t + W_h * h_prev + b)
    return np.tanh(x_t @ Wx + h_prev @ Wh + b)

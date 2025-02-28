import numba
import numpy as np
from numba import njit, prange
from neura_nova.init import glorot_uniform_init_conv

numba.set_num_threads(6)

@njit(parallel=True)
def conv_forward_naive(X_padded, weights, bias, stride, kernel_size, filter_number, out_h, out_w):
    """
    Esegue la convoluzione 'naive' con loop annidati su batch, filtri, out_h e out_w.

    Parametri:
    ----------
    X_padded      : np.array di shape (batch_size, in_channels, H_pad, W_pad)
    weights       : np.array di shape (filter_number, in_channels, kernel_size, kernel_size)
    bias          : np.array di shape (filter_number, 1)
    stride        : int
    kernel_size   : int
    filter_number : int
    out_h, out_w  : dimensioni spaziali dell'output

    Ritorno:
    --------
    output : np.array di shape (batch_size, filter_number, out_h, out_w)
    """
    batch_size, _, _, _ = X_padded.shape
    filter_number = weights.shape[0]

    out = np.zeros((batch_size, filter_number, out_h, out_w), dtype=X_padded.dtype)
    for b in prange(batch_size):
        for f in range(filter_number):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride
                    h_end   = h_start + kernel_size
                    w_start = j * stride
                    w_end   = w_start + kernel_size
                    patch   = X_padded[b, :, h_start:h_end, w_start:w_end]
                    out[b, f, i, j] = np.sum(patch * weights[f]) + bias[f, 0]
    return out

@njit(parallel=True)
def conv_backward_naive(X_padded, weights, grad_output, stride, kernel_size):
    """
    Calcola:
      - grad_weights
      - grad_bias
      - grad_input_padded
    con un approccio naive: quadruplo for (batch, filter, i, j).

    Parametri:
    ----------
    X_padded    : np.array (batch_size, in_channels, H_pad, W_pad)
    weights     : np.array (filter_number, in_channels, kernel_size, kernel_size)
    grad_output : np.array (batch_size, filter_number, out_height, out_width)
    stride      : int
    kernel_size : int

    Ritorna:
    --------
    grad_input_padded : np.array (stessa shape di X_padded)
    grad_weights      : np.array (stessa shape di weights)
    grad_bias         : np.array (shape = (filter_number, 1))
    """
    batch_size, in_channels, H_pad, W_pad = X_padded.shape
    filter_number = weights.shape[0]
    out_height    = grad_output.shape[2]
    out_width     = grad_output.shape[3]

    grad_weights      = np.zeros_like(weights)
    grad_bias         = np.zeros((filter_number, 1), dtype=weights.dtype)
    grad_input_padded = np.zeros_like(X_padded)

    # Manualmente: somma su (batch, i, j) per ogni f
    for b in prange(batch_size):
        for f in range(filter_number):
            for i in range(out_height):
                for j in range(out_width):
                    grad_bias[f, 0] += grad_output[b, f, i, j]

            # Calcolo di grad_weights e grad_input_padded
            for i in range(out_height):
                for j in range(out_width):
                    gval    = grad_output[b, f, i, j]
                    h_start = i * stride
                    w_start = j * stride

                    for c in range(in_channels):
                        for hh in range(kernel_size):
                            for ww in range(kernel_size):
                                grad_weights[f, c, hh, ww] += X_padded[b, c, h_start+hh, w_start+ww] * gval

                    # grad_input_padded aggiunge:
                    # grad_input_padded[b, :, h_start:h_end, w_start:w_end] += weights[f]*gval
                    for c in range(in_channels):
                        for hh in range(kernel_size):
                            for ww in range(kernel_size):
                                grad_input_padded[b, c, h_start+hh, w_start+ww] += weights[f, c, hh, ww] * gval
    return grad_input_padded, grad_weights, grad_bias

def compute_padding(in_dim, kernel_size, stride):
    # Se il kernel è pari e stride = 1, non esiste una soluzione "perfetta"
    # simmetrica e intera

    # Per stride = 1
    # Usare kernel dispari (1, 3, 5, 7, ...) --> pad = (k-1)/2

    # Per stride > 1 (2, 4)
    # 2, 4, 6

    # per ottenere out_dim = (in_dim + 2*pad - kernel_size) // stride + 1 == in_dim

    pad_tot = stride * (in_dim - 1) + kernel_size - in_dim
    return max(0, pad_tot // 2)


class Conv2D:
    def __init__(self, input_channels, filter_number, kernel_size, stride, activation_funct, learning_rate,
                 beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.input_channels   = input_channels
        self.filter_number    = filter_number
        self.kernel_size      = kernel_size
        self.stride           = stride
        self.activation_funct = activation_funct

        self.weights = None
        self.bias    = None
        self.weights = np.ascontiguousarray(self.weights, dtype=np.float32)
        self.bias    = np.ascontiguousarray(self.bias, dtype=np.float32)

        # self.weights shape: (filters, input_channels, kernel_size, kernel_size)
        # self.bias    shape: (self.num_filters, 1)
        if activation_funct == 'relu':
            # Kaiming/He initialization
            self.weights = (
                    np.random.randn(self.filter_number, self.input_channels, self.kernel_size, self.kernel_size) *
                    np.sqrt(2. / (self.input_channels * self.kernel_size * self.kernel_size))
            ).astype(np.float32)
        else:
            # Xavier/Glorot initialization
            self.weights = glorot_uniform_init_conv(self.filter_number, self.input_channels, self.kernel_size)
        self.bias = np.zeros((self.filter_number, 1), dtype=np.float32)

        # Parametri per Adam
        self.learning_rate = learning_rate
        self.beta1         = beta1
        self.beta2         = beta2
        self.epsilon       = epsilon
        self.t             = 0

        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_bias    = np.zeros_like(self.bias)
        self.v_bias    = np.zeros_like(self.bias)

        # Cache
        self.input             = None  # input originale
        self.X_padded          = None
        self.output            = None
        self.activation_cache  = None
        self.out_h             = None
        self.out_w             = None

    def get_weights(self):
        return (
            np.copy(self.weights), np.copy(self.bias),
            np.copy(self.m_weights), np.copy(self.v_weights),
            np.copy(self.m_bias), np.copy(self.v_bias),
            self.t
        )

    def set_weights(self, saved_state):
        (
            self.weights, self.bias,
            self.m_weights, self.v_weights,
            self.m_bias, self.v_bias,
            self.t
        ) = saved_state

    def forward(self, X):
        self.input = X
        batch_size, _, H, W = X.shape

        pad_h = compute_padding(H, self.kernel_size, self.stride)
        pad_w = compute_padding(W, self.kernel_size, self.stride)

        self.X_padded = np.pad(X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')

        # Calculate output dimensions
        self.out_h = (H + 2 * pad_h - self.kernel_size) // self.stride + 1
        self.out_w = (W + 2 * pad_w - self.kernel_size) // self.stride + 1

        # Per ottenere "same shape" (kernel_size dispari, stride, padding)
        # (k = 1, p = 0, s = 1)
        # (k = 3, p = 1, s = 1)
        # (k = 5, p = 2, s = 1)
        # (k = 7, p = 3, s = 1)

        # Per ottenere "same shape" (kernel_size pari, stride, padding)
        # (k = 2, p = 14, s = 2)

        if self.out_h != H or self.out_w != W:
            print("Wrong choice of (kernel_size, stride, padding)")
            raise ValueError("[CONVOLUTION] Constraint not satisfied: out_h = H, out_w = W")

        # Initialize output
        output = np.zeros((batch_size, self.filter_number, self.out_h, self.out_w))

        # Perform convolution
        output = conv_forward_naive(
            self.X_padded,
            self.weights,
            self.bias,
            self.stride,
            self.kernel_size,
            self.filter_number,
            self.out_h,
            self.out_w
        )

        if self.activation_funct == 'relu':
            self.activation_cache = (output > 0).astype(np.float32)
            output = np.maximum(0, output)
        elif self.activation_funct == 'sigmoid':
            output = 1 / (1 + np.exp(-output))
            self.activation_cache = output * (1 - output)
        else:
            self.activation_cache = np.ones_like(output)

        self.output = output
        return output

    def backward(self, grad_output):
        # grad_output shape: (batch_size, filter_number, out_height, out_width)
        if self.output is None:
            raise ValueError("Must call forward before backward")

        self.t += 1
        batch_size, _, out_height, out_width = grad_output.shape

        # Apply activation function derivative
        if self.activation_funct == 'relu':
            grad_output = grad_output * self.activation_cache
        elif self.activation_funct == 'sigmoid':
            grad_output = grad_output * self.activation_cache

        grad_input_padded, grad_weights, grad_bias = conv_backward_naive(
            self.X_padded,   # shape: (batch_size, in_channels, H_pad, W_pad)
            self.weights,    # shape: (filter_number, in_channels, kernel_size, kernel_size)
            grad_output,     # shape: (batch_size, filter_number, out_height, out_width)
            self.stride,
            self.kernel_size
        )

        pad_h = (self.X_padded.shape[2] - self.input.shape[2]) // 2
        pad_w = (self.X_padded.shape[3] - self.input.shape[3]) // 2

        # Rimuovere padding da grad_input
        if pad_h > 0 or pad_w > 0:
            grad_input = grad_input_padded[:, :, pad_h:-pad_h, pad_w:-pad_w]
        else:
            grad_input = grad_input_padded

        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * grad_weights
        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (grad_weights ** 2)
        self.m_bias    = self.beta1 * self.m_bias    + (1 - self.beta1) * grad_bias
        self.v_bias    = self.beta2 * self.v_bias    + (1 - self.beta2) * (grad_bias ** 2)

        m_hat_weights = self.m_weights / (1 - self.beta1 ** self.t)
        v_hat_weights = self.v_weights / (1 - self.beta2 ** self.t)
        m_hat_bias    = self.m_bias    / (1 - self.beta1 ** self.t)
        v_hat_bias    = self.v_bias    / (1 - self.beta2 ** self.t)

        self.weights -= self.learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)
        self.bias    -= self.learning_rate * m_hat_bias    / (np.sqrt(v_hat_bias)    + self.epsilon)

        return grad_input
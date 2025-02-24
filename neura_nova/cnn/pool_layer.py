import numba
import numpy as np
from numba import njit, prange

numba.set_num_threads(6)

@njit(parallel=True)
def maxpool_forward_naive(X, kernel_size, stride):
    """
    Esecuzione naive del max pooling con loop, parallelizzata su batch (e canali).
    Ritorna:
      - output: shape (batch_size, channels, out_h, out_w)
      - argmax: shape (batch_size, channels, out_h, out_w, 2),
                per salvare la posizione (hh, ww) del massimo nella finestra.
    """
    batch_size, channels, H, W = X.shape
    out_h = (H - kernel_size) // stride + 1
    out_w = (W - kernel_size) // stride + 1

    output = np.zeros((batch_size, channels, out_h, out_w), dtype=X.dtype)
    argmax = np.zeros((batch_size, channels, out_h, out_w, 2), dtype=np.int32)

    # Parallelizziamo sul batch e sui canali in un unico loop
    for bc in prange(batch_size * channels):
        b = bc // channels
        c = bc % channels
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                w_start = j * stride
                max_val = -np.inf
                max_i   = 0
                max_j   = 0

                # Ricerca del massimo in kernel_size x kernel_size
                for hh in range(kernel_size):
                    for ww in range(kernel_size):
                        val = X[b, c, h_start + hh, w_start + ww]
                        if val > max_val:
                            max_val = val
                            max_i   = hh
                            max_j   = ww
                output[b, c, i, j]    = max_val
                argmax[b, c, i, j, 0] = max_i
                argmax[b, c, i, j, 1] = max_j
    return output, argmax

@njit(parallel=True)
def maxpool_backward_naive(grad_output, argmax, kernel_size, stride, in_shape):
    """
    Backprop del Max Pooling.
    grad_output: (batch_size, channels, out_h, out_w)
    argmax     : (batch_size, channels, out_h, out_w, 2)
    in_shape   : (batch_size, channels, H, W) = shape dell'input originale

    Ritorna:
      dinput : gradiente w.r.t. input, shape = in_shape
    """
    batch_size, channels, H, W = in_shape
    out_h, out_w = grad_output.shape[2], grad_output.shape[3]

    dinput = np.zeros(in_shape, dtype=grad_output.dtype)

    for bc in prange(batch_size * channels):
        b = bc // channels
        c = bc % channels
        for i in range(out_h):
            for j in range(out_w):
                grad_val = grad_output[b, c, i, j]
                local_i  = argmax[b, c, i, j, 0]
                local_j  = argmax[b, c, i, j, 1]
                h_start  = i * stride
                w_start  = j * stride

                dinput[b, c, h_start + local_i, w_start + local_j] += grad_val
    return dinput

class MaxPoolLayer:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride      = stride
        self.input       = None
        self.argmax      = None

    def forward(self, X):
        self.input   = X
        output, argm = maxpool_forward_naive(X, self.kernel_size, self.stride)
        self.argmax  = argm
        return output

    def backward(self, grad_output):
        batch_size, channels, H, W = self.input.shape
        dinput = maxpool_backward_naive(
            grad_output,
            self.argmax,
            self.kernel_size,
            self.stride,
            (batch_size, channels, H, W)
        )
        return dinput
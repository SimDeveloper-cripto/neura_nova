import numpy as np

# TODO: LO SPESSORE DEL FEATURE VOLUME RISULTANTE E' PARI AL NUMERO DI FILTRI?
class MaxPoolLayer:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride      = stride
        self.input       = None
        self.argmax      = None
        self.out_h       = None
        self.out_w       = None

    def forward(self, X):
        self.input = X
        batch_size, channels, H, W = X.shape

        self.out_h = (H - self.kernel_size) // self.stride + 1
        self.out_w = (W - self.kernel_size) // self.stride + 1

        output = np.zeros((batch_size, channels, self.out_h, self.out_w), dtype=X.dtype)

        # 2 --> Salviamo indici (riga, colonna) di dove si trova il massimo
        self.argmax = np.zeros((batch_size, channels, self.out_h, self.out_w, 2), dtype=np.int32)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(self.out_h):
                    for j in range(self.out_w):
                        h_start = i * self.stride
                        h_end   = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end   = w_start + self.kernel_size

                        patch   = X[b, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(patch)
                        output[b, c, i, j] = max_val

                        local_i, local_j = np.unravel_index(np.argmax(patch), patch.shape)
                        self.argmax[b, c, i, j] = (local_i, local_j)
        return output

    def backward(self, grad_output):
        batch_size, channels, out_h, out_w = grad_output.shape
        _, _, H, W = self.input.shape
        dinput = np.zeros_like(self.input)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        # Gradiente relativo al singolo valore max nella finestra
                        grad_val = grad_output[b, c, i, j]

                        # Posizione locale del massimo
                        local_i, local_j = self.argmax[b, c, i, j]

                        h_start = i * self.stride
                        w_start = j * self.stride

                        # Aggiungiamo il gradiente solo nella posizione di max
                        dinput[b, c, h_start + local_i, w_start + local_j] += grad_val
        return dinput
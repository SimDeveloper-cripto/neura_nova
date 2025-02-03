# neura_nova/conv_layer.py

import numpy as np

class ConvLayer:
    # Convolutional neural network layer

    """ [NOTA]
    In un layer convoluzionale il neurone non corrisponde a un’unità isolata come nei layer fully-connected, ma a ciascuna posizione spaziale nella feature map in output.
    Se l'input (immagine) ha dimensioni H×W e viene usata una convoluzione con filtro 3×3, stride 1 e padding 1, l’output manterrà le stesse dimensioni spaziali (H×W).
    Quindi, per ciascuno dei filtri (ciascuno dei quali “scansiona” l’immagine con la sua finestra 3×3), si ottiene una feature map formata da H×W “neuroni” (cioè da H×W uscite).
    Il numero totale di neuroni nel layer sarà dunque (# di filtri) × (H * W).

    Per ora i filtri non sembrano cercare dei pattern specifici.
    Con il passare delle epoche, i filtri imparano a estrarre le caratteristiche rilevanti presenti nei dati, come bordi, texture o forme complesse,
    in modo da minimizzare l'errore della rete sul compito specifico (ovvero la classificazione delle immagini).
    """

    def __init__(self, input_channels, activation='relu',
                 learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 manual_filters=False):
        self.input_channels = input_channels
        self.num_filters    = 6
        self.kernel_size    = 3
        self.stride         = 1
        self.padding        = 1
        self.manual_filters = manual_filters
        self.activation     = activation

        self.weights = None
        self.bias    = None
        self.weights = np.ascontiguousarray(self.weights, dtype=np.float32)
        self.bias    = np.ascontiguousarray(self.bias, dtype=np.float32)

        if manual_filters:
            base_filters = [
                np.array([[1,  0, -1],
                          [2,  0, -2],
                          [1,  0, -1]], dtype=np.float32),  # Vertical lines
                np.array([[1,  2,  1],
                          [0,  0,  0],
                          [-1, -2, -1]], dtype=np.float32),  # Horizontal lines
                np.array([[2,  1,  0],
                          [1,  0, -1],
                          [0, -1, -2]], dtype=np.float32),  # 45° diagonal
                np.array([[0,  1,  2],
                          [-1, 0,  1],
                          [-2, -1,  0]], dtype=np.float32),  # 135° diagonal
                np.array([[0,  1,  0],
                          [1, -4,  1],
                          [0,  1,  0]], dtype=np.float32),  # Curves (Laplacian)
                np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]], dtype=np.float32)   # Angles
            ]
            self.weights = np.zeros((self.num_filters, self.input_channels, self.kernel_size, self.kernel_size), dtype=np.float32)

            for f in range(self.num_filters):
                for c in range(self.input_channels):
                    self.weights[f, c, :, :] = base_filters[f]
            self.bias = np.zeros((6, 1), dtype=np.float32)

            # Memoria per Adam (non verranno però aggiornate)
            self.m_weights = np.zeros_like(self.weights)
            self.v_weights = np.zeros_like(self.weights)
            self.m_bias    = np.zeros_like(self.bias)
            self.v_bias    = np.zeros_like(self.bias)
        else:
            if activation == 'relu':
                # Kaiming/He initialization
                self.weights = np.random.randn(self.num_filters, self.input_channels, self.kernel_size, self.kernel_size) *\
                               np.sqrt(2. / (self.input_channels * self.kernel_size * self.kernel_size))
            else:
                # Xavier/Glorot initialization
                self.weights = np.random.randn(self.num_filters, self.input_channels, self.kernel_size, self.kernel_size) * \
                               np.sqrt(1. / (self.input_channels * self.kernel_size * self.kernel_size))
                # self.weights shape: (6, 1, 3, 3)
            self.bias = np.zeros((self.num_filters, 1), dtype=np.float32)

            # Parametri per Adam
            self.learning_rate = learning_rate
            self.beta1         = beta1
            self.beta2         = beta2
            self.epsilon       = epsilon
            self.t             = 0  # step counter

            self.m_weights = np.zeros_like(self.weights)
            self.v_weights = np.zeros_like(self.weights)
            self.m_bias    = np.zeros_like(self.bias)
            self.v_bias    = np.zeros_like(self.bias)

        # Variabili per forward/backward
        self.input            = None  # input originale in forma (batch_size, input_channels, H, W)
        self.input_padded     = None  # input dopo il padding
        self.output           = None
        self.activation_cache = None
        self.pre_activation   = None

    def forward(self, input_data):
        """
        :param input_data: (batch_size, input_channels, H, W)
        :return: (batch_size, num_filters, H_out, W_out)
        """
        self.input = input_data
        batch_size, _, H, W = input_data.shape

        pad = self.padding
        self.input_padded = np.pad(input_data,
                                   pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)),
                                   mode='constant', constant_values=0)
        H_padded, W_padded = H + 2*pad, W + 2*pad

        # Dimensioni di output
        H_out = int((H_padded - self.kernel_size) / self.stride) + 1
        W_out = int((W_padded - self.kernel_size) / self.stride) + 1

        output = np.zeros((batch_size, self.num_filters, H_out, W_out), dtype=np.float32)

        # Convoluzione: per ogni immagine del batch, per ogni filtro e per ogni posizione
        for n in range(batch_size):
            for f in range(self.num_filters):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * self.stride
                        h_end   = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end   = w_start + self.kernel_size

                        # shape: (input_channels, kernel_size, kernel_size)
                        patch = self.input_padded[n, :, h_start:h_end, w_start:w_end]

                        # self.weights[f] shape: (1, 3, 3) --> (input_channels, kernel_size, kernel_size)
                        output[n, f, i, j] = np.sum(self.weights[f] * patch) + self.bias[f]

        self.pre_activation = output.copy()
        match self.activation:
            case 'relu':
                output = np.maximum(0, output)
                self.activation_cache = (output > 0).astype(np.float32)
            case 'sigmoid':
                output = 1 / (1 + np.exp(-output))
                self.activation_cache = self.output * (1 - self.output)
            case 'identity':
                output = output
                self.activation_cache = np.ones_like(output)
            case _:
                raise ValueError(f"Unsupported activation function: {self.activation}")

        self.output = output
        return output

    def backward(self, grad_output):
        batch_size, _, H_out, W_out = grad_output.shape
        _, _, H_padded, W_padded    = self.input_padded.shape

        grad_z = grad_output * self.activation_cache

        # Gradienti per i pesi e il bias
        dW = np.zeros_like(self.weights, dtype=np.float32)
        dB = np.zeros_like(self.bias, dtype=np.float32)

        d_input_padded = np.zeros_like(self.input_padded, dtype=np.float32)

        for n in range(batch_size):
            for f in range(self.num_filters):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * self.stride
                        h_end   = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end   = w_start + self.kernel_size

                        patch = self.input_padded[n, :, h_start:h_end, w_start:w_end]
                        dW[f] += grad_z[n, f, i, j] * patch
                        dB[f] += grad_z[n, f, i, j]

                        # Quadrato 3x3 in analisi
                        d_input_padded[n, :, h_start:h_end, w_start:w_end] += self.weights[f] * grad_z[n, f, i, j]

        # Scartiamo il padding aggiunto artificiosamente durante il forward
        # Il gradiente restituito corrisponde alle dimensioni originali dell'input
        if self.padding > 0:
            d_input = d_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            d_input = d_input_padded

        if not self.manual_filters:
            self.t += 1

            self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * dW
            self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (dW ** 2)
            m_hat_weights  = self.m_weights / (1 - self.beta1 ** self.t)
            v_hat_weights  = self.v_weights / (1 - self.beta2 ** self.t)
            self.weights   -= self.learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)

            self.m_bias = self.beta1 * self.m_bias + (1 - self.beta1) * dB
            self.v_bias = self.beta2 * self.v_bias + (1 - self.beta2) * (dB ** 2)
            m_hat_bias  = self.m_bias / (1 - self.beta1 ** self.t)
            v_hat_bias  = self.v_bias / (1 - self.beta2 ** self.t)
            self.bias   -= self.learning_rate * m_hat_bias / (np.sqrt(v_hat_bias) + self.epsilon)

        return d_input


class MaxPoolLayer:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride      = stride
        self.input       = None
        self.mask        = None  # Mask for position of the maximum value

    # TODO: DA IMPLEMENTARE
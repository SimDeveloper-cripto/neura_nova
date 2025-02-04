import numpy as np


def im2col(input_data, kernel_size, stride, padding):
    """
    Trasforma l'input (batch_size, channels, H, W) in una matrice 2D, in cui ogni riga
    corrisponde a una patch (flattened) dell'immagine.

    Returns:
        - cols: array of shape (batch_size * out_h, out_w, channels * kernel_size * kernel_size)
        - out_h, out_w
    """
    batch_size, channels, H, W = input_data.shape

    input_padded       = np.pad(input_data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    out_h              = (H_padded - kernel_size) // stride + 1
    out_w              = (W_padded - kernel_size) // stride + 1

    shape = (batch_size, channels, out_h, out_w, kernel_size, kernel_size)
    strides = (input_padded.strides[0],
               input_padded.strides[1],
               stride * input_padded.strides[2],
               stride * input_padded.strides[3],
               input_padded.strides[2],
               input_padded.strides[3])
    patches = np.lib.stride_tricks.as_strided(input_padded, shape=shape, strides=strides)

    # Each row corresponds to a flattened patch
    cols = patches.reshape(batch_size, channels, out_h * out_w, kernel_size, kernel_size)
    cols = cols.transpose(0, 2, 1, 3, 4).reshape(batch_size * out_h * out_w, -1)
    return cols, out_h, out_w

def col2im(cols, input_shape, kernel_size, stride, padding, out_h, out_w):
    """
    - input_shape: (batch_size, channels, H, W)
    - Returns an array of the same shape as 'input_data'.
    """
    batch_size, channels, H, W = input_shape
    H_padded, W_padded         = H + 2 * padding, W + 2 * padding
    input_padded               = np.zeros((batch_size, channels, H_padded, W_padded), dtype=cols.dtype)

    # Reshape cols
    cols_reshaped = cols.reshape(batch_size, out_h * out_w, channels, kernel_size, kernel_size)
    cols_reshaped = cols_reshaped.transpose(0, 2, 1, 3, 4)  # (batch_size, channels, out_h*out_w, kernel_size, kernel_size)

    # Sum patches relative to each position
    idx = 0
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            h_end   = h_start + kernel_size
            w_start = j * stride
            w_end   = w_start + kernel_size
            input_padded[:, :, h_start:h_end, w_start:w_end] += cols_reshaped[:, :, idx, :, :]
            idx += 1

    # Remove padding
    if padding > 0:
        return input_padded[:, :, padding:-padding, padding:-padding]
    return input_padded


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

    def __init__(self, input_channels, num_filters, kernel_size, stride, padding, activation='relu',
                 learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.input_channels = input_channels
        self.num_filters    = num_filters
        self.kernel_size    = kernel_size
        self.stride         = stride
        self.padding        = padding
        self.activation     = activation

        self.weights = None
        self.bias    = None
        self.weights = np.ascontiguousarray(self.weights, dtype=np.float32)
        self.bias    = np.ascontiguousarray(self.bias, dtype=np.float32)

        if activation == 'relu':
            # Kaiming/He initialization
            self.weights = np.random.randn(self.num_filters, self.input_channels, self.kernel_size, self.kernel_size) * \
                           np.sqrt(2. / (self.input_channels * self.kernel_size * self.kernel_size))
        else:
            # Xavier/Glorot initialization
            self.weights = np.random.randn(self.num_filters, self.input_channels, self.kernel_size, self.kernel_size) * \
                           np.sqrt(1. / (self.input_channels * self.kernel_size * self.kernel_size))
            # self.weights shape: (filters, input_channels, kernel_size, kernel_size)
        self.bias = np.zeros((self.num_filters, input_channels), dtype=np.float32)

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
        self.cols  = None
        self.out_h = None
        self.out_w = None

    def forward(self, input_data):
        """
        :param input_data: (batch_size, input_channels, H, W)
        :return: (batch_size, num_filters, H_out, W_out)
        """
        self.input = input_data
        batch_size, _, H, W = input_data.shape

        cols, out_h, out_w = im2col(input_data, self.kernel_size, self.stride, self.padding)
        self.cols  = cols
        self.out_h = out_h
        self.out_w = out_w

        # Organize filters in a column matrix
        filters_col = self.weights.reshape(self.num_filters, -1).T  # shape: (channels * kernel_size^2, num_filters)
        conv = cols.dot(filters_col) + self.bias.ravel()            # shape: (batch_size*out_h*out_w, num_filters)

        # Re-organize output
        conv = conv.reshape(batch_size, out_h, out_w, self.num_filters).transpose(0, 3, 1, 2)
        self.pre_activation = conv.copy()

        match self.activation:
            case 'relu':
                output = np.maximum(0, conv)
                self.activation_cache = (conv > 0).astype(np.float32)
            case 'sigmoid':
                output = 1 / (1 + np.exp(-conv))
                self.activation_cache = output * (1 - output)
            case 'identity':
                output = conv
                self.activation_cache = np.ones_like(conv)
            case _:
                raise ValueError(f"Unsupported activation function: {self.activation}")

        self.output = output
        return output

    def backward(self, grad_output):
        # grad_output shape: (batch_size, num_filters, out_h, out_w)
        batch_size = self.input.shape[0]

        dconv = grad_output * self.activation_cache                                 # shape: (batch_size, num_filters, out_h, out_w)
        dconv_reshaped = dconv.transpose(0, 2, 3, 1).reshape(-1, self.num_filters)  # (batch_size*out_h*out_w, num_filters)

        # shape: (channels*kernel_size^2, num_filters)
        dW = self.cols.T.dot(dconv_reshaped)
        dW = dW.transpose(1, 0).reshape(self.weights.shape)

        # shape: (num_filters, 1)
        dB = np.sum(dconv_reshaped, axis=0, keepdims=True).T

        filters_col = self.weights.reshape(self.num_filters, -1).T
        dcols = dconv_reshaped.dot(filters_col.T)
        dx = col2im(dcols, self.input.shape, self.kernel_size, self.stride, self.padding, self.out_h, self.out_w)

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

        return dx

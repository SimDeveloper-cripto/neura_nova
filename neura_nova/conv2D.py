import numpy as np
from neura_nova.init import glorot_uniform_init_conv


class Conv2D:
    def __init__(self, input_channels, filter_number, kernel_size, stride, padding, activation_funct, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.input_channels     = input_channels
        self.filter_number      = filter_number
        self.kernel_size        = kernel_size
        self.stride             = stride
        self.padding            = padding
        self.activation_funct   = activation_funct
        self.kernels            = np.random.randn(filter_number, input_channels, kernel_size, kernel_size)

        self.weights = None
        self.bias = None
        self.weights = np.ascontiguousarray(self.weights, dtype=np.float32)
        self.bias = np.ascontiguousarray(self.bias, dtype=np.float32)

        # self.weights shape: (filters, input_channels, kernel_size, kernel_size)
        # self.bias    shape: mono-dimensional array of dim (self.num_filters,)
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
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # step counter

        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_bias = np.zeros_like(self.bias)
        self.v_bias = np.zeros_like(self.bias)

        # Variabili per forward/backward
        self.input            = None  # input originale in forma (batch_size, input_channels, H, W)
        self.output           = None
        self.activation_cache = None
        self.pre_activation   = None
        self.cols  = None
        self.out_h = None
        self.out_w = None

        # Cache for backward pass
        self.input = None
        self.output = None
        self.X_padded = None

    def forward(self, X):
        """Forward pass for convolutional layer"""
        self.input = X
        batch_size, _, H, W = X.shape

        # Calculate output dimensions
        self.out_h = (H - self.kernel_size + 2 * self.padding) // self.stride + 1
        self.out_w = (W - self.kernel_size + 2 * self.padding) // self.stride + 1

        # Pad input and store it
        self.X_padded = np.pad(X,
                               ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                               mode='constant')

        # Initialize output
        output = np.zeros((batch_size, self.filter_number, self.out_h, self.out_w))

        # Perform convolution
        for b in range(batch_size):
            for f in range(self.filter_number):
                for i in range(self.out_h):
                    for j in range(self.out_w):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        patch = self.X_padded[b, :, h_start:h_end, w_start:w_end]
                        output[b, f, i, j] = np.sum(patch * self.weights[f]) + self.bias[f]

        # Apply activation function
        if self.activation_funct == 'relu':
            self.activation_cache = (output > 0).astype(np.float32)
            output = np.maximum(0, output)
        elif self.activation_funct == 'sigmoid':
            output = 1 / (1 + np.exp(-output))
            self.activation_cache = output * (1 - output)
        else:  # identity
            self.activation_cache = np.ones_like(output)

        self.output = output
        return output

    def backward(self, grad_output):
        """
        Backward pass for convolutional layer
        grad_output shape: (batch_size, filter_number, out_height, out_width)
        """
        if self.output is None:
            raise ValueError("Must call forward before backward")

        self.t += 1
        batch_size, _, out_height, out_width = grad_output.shape

        # Apply activation function derivative
        if self.activation_funct == 'relu':
            grad_output = grad_output * self.activation_cache
        elif self.activation_funct == 'sigmoid':
            grad_output = grad_output * self.activation_cache

        # Initialize gradients
        grad_weights = np.zeros_like(self.weights)
        grad_bias = np.sum(grad_output, axis=(0, 2, 3), keepdims=True).T
        grad_input_padded = np.zeros_like(self.X_padded)

        # Calculate gradients
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                patch = self.X_padded[:, :, h_start:h_end, w_start:w_end]

                for k in range(self.filter_number):
                    # Gradient of weights
                    grad_weights[k] += np.sum(
                        patch * grad_output[:, k, i, j][:, None, None, None],
                        axis=0
                    )

                    # Gradient of input
                    grad_input_padded[:, :, h_start:h_end, w_start:w_end] += \
                        self.weights[k] * grad_output[:, k, i, j][:, None, None, None]

        # Remove padding from gradient
        if self.padding > 0:
            grad_input = grad_input_padded[:, :,
                         self.padding:-self.padding,
                         self.padding:-self.padding
                         ]
        else:
            grad_input = grad_input_padded

        # Update weights and biases using Adam
        # Momentum update
        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * grad_weights
        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (grad_weights ** 2)
        self.m_bias = self.beta1 * self.m_bias + (1 - self.beta1) * grad_bias
        self.v_bias = self.beta2 * self.v_bias + (1 - self.beta2) * (grad_bias ** 2)

        # Bias correction
        m_hat_weights = self.m_weights / (1 - self.beta1 ** self.t)
        v_hat_weights = self.v_weights / (1 - self.beta2 ** self.t)
        m_hat_bias = self.m_bias / (1 - self.beta1 ** self.t)
        v_hat_bias = self.v_bias / (1 - self.beta2 ** self.t)

        # Update parameters
        self.weights -= self.learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)
        self.bias -= self.learning_rate * m_hat_bias / (np.sqrt(v_hat_bias) + self.epsilon)

        return grad_input


if __name__ == "__main__":
    # Testiamo con un batch fittizio di immagini
    batch_size = 2
    in_channels = 3
    height = 32
    width = 32
    input_data = np.random.randn(batch_size, in_channels, height, width)

    # Creiamo il layer convoluzionale con 6 filtri 3x3, stride 1, padding 1
    conv = Conv2D(
        input_channels=1,
        filter_number=16,
        kernel_size=3,
        stride=1,
        padding=1,
        activation_funct='relu',
        learning_rate=0.001
    )

    # Forward pass
    output = conv.forward(input_data)

    # Print shapes to verify
    print("Input shape:", input_data.shape)
    print("Output shape:", output.shape)
    print("out_h:", conv.out_h)
    print("out_w:", conv.out_w)

    # Backward pass
    grad_output = np.random.randn(*output.shape)  # Create random gradient
    grad_input = conv.backward(grad_output.reshape(output.shape[0], -1).T)

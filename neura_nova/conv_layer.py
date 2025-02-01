# neura_nova/conv_layer.py

import numpy as np

class ConvLayer:
    # Convolutional neural network layer
    def __init__(self, input_channels, num_filters, kernel_size=3, stride=1, padding=1,
                 activation='relu', learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        :param input_channels: number of input channels (depth of input)
        :param num_filters: number of convolutional filters
        :param kernel_size: size of the convolutional kernel (square is assumed)
        :param stride: stride of the convolution
        :param padding: padding added to the input on all sides
        :param activation: activation function
        """
        self.input_channels = input_channels
        self.num_filters    = num_filters
        self.kernel_size    = kernel_size
        self.stride         = stride
        self.padding        = padding
        self.activation     = activation

        # Parametri per Adam
        self.learning_rate = learning_rate
        self.beta1         = beta1
        self.beta2         = beta2
        self.epsilon       = epsilon
        self.t             = 0  # step counter

        self.weights = None
        if activation == 'relu':
            # Kaiming/He initialization
            self.weights = np.random.randn(num_filters, input_channels, kernel_size, kernel_size) *\
                           np.sqrt(2. / (input_channels * kernel_size * kernel_size))
        else:
            # Xavier/Glorot initialization
            self.weights = np.random.randn(num_filters, input_channels, kernel_size, kernel_size) * \
                           np.sqrt(1. / (input_channels * kernel_size * kernel_size))
        self.bias = np.zeros((num_filters, 1))

        self.weights = np.ascontiguousarray(self.weights, dtype=np.float32)
        self.bias    = np.ascontiguousarray(self.bias, dtype=np.float32)

        # Memoria per Adam
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_bias    = np.zeros_like(self.bias)
        self.v_bias    = np.zeros_like(self.bias)

        # Variabili per forward/backward
        self.input_shape      = None  # forma originale dell'input (N, C, H, W)
        self.X_col            = None  # output di im2col
        self.Z                = None  # output lineare (prima dell'attivazione)
        self.output           = None  # output dopo attivazione
        self.activation_cache = None  # cache per il gradiente dell'attivazione
        self.out_h            = None  # altezza dell'output
        self.out_w            = None  # larghezza dell'output










    def forward(self, input_data):
        # TODO: DA IMPLEMENTARE
        pass

    def backward(self, grad_output, learning_rate):
        # TODO: DA IMPLEMENTARE
        pass
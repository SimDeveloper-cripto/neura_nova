# neura_nova/conv_layer.py

import numpy as np

class ConvLayer:
    # Convolutional neural network layer
    def __init__(self, input_channels, num_filters, kernel_size=3, stride=1, padding=1, activation='relu'):
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

        self.input            = None
        self.output           = None
        self.activation_cache = None

        if activation == 'relu':
            # Kaiming/He initialization
            self.weights = np.random.randn(num_filters, input_channels, kernel_size, kernel_size) *\
                           np.sqrt(2. / (input_channels * kernel_size * kernel_size))
        else:
            # Xavier/Glorot initialization
            self.weights = np.random.randn(num_filters, input_channels, kernel_size, kernel_size) * \
                           np.sqrt(1. / (input_channels * kernel_size * kernel_size))
        self.bias = np.zeros((num_filters, 1))

    def forward(self, input_data):
        #TODO: DA IMPLEMENTARE
        pass

    def backward(self, grad_output, learning_rate):
        # TODO: DA IMPLEMENTARE
        pass
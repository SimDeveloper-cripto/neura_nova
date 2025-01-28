# neura_nova/layers.py

import numpy as np


class DenseLayer:
    # A fully connected neural network layer
    def __init__(self, input_dim, output_dim, activation='relu'):
        """
        :param input_dim : number of input features
        :param output_dim: number of neurons in the layer
        :param activation: activation function
        """
        # TODO: FIX RANDOM INIT
        self.weights    = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.bias       = np.zeros((1, output_dim))
        self.input      = None
        self.output     = None
        self.activation = activation
        self.activation_cache = None  # Derivative of activation

    def forward(self, input_data):
        self.input = input_data

        # TODO: Z = WX + B
        z = np.dot(input_data, self.weights) + self.bias

        match self.activation:
            case 'relu':
                self.output = np.maximum(0, z)
                self.activation_cache = z > 0
            case 'sigmoid':
                self.output = 1 / (1 + np.exp(-z))
                self.activation_cache = self.output * (1 - self.output)
            case 'identity':
                self.output = z
                self.activation_cache = 1
            case _:
                raise ValueError(f"Unsupported activation function: {self.activation}")
        return self.output

    def backward(self, grad_output, learning_rate):
        if self.activation == 'relu' or self.activation == 'sigmoid'\
                or self.activation == 'identity':
            grad_z = grad_output * self.activation_cache
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        grad_weights = np.dot(self.input.T, grad_z)           # shape (input_dim, output_dim)
        grad_bias    = np.sum(grad_z, axis=0, keepdims=True)  # shape (1, output_dim)
        grad_input   = np.dot(grad_z, self.weights.T)

        # Update network parameters
        self.weights -= learning_rate * grad_weights
        self.bias    -= learning_rate * grad_bias
        return grad_input

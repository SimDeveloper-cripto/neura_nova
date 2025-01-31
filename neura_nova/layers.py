# neura_nova/layers.py

import numpy as np


class DenseLayer:
    # A fully connected neural network layer
    def __init__(self, input_dim, output_dim, activation='relu'):
        """
        :param input_dim : number of input features/neurons
        :param output_dim: number of neurons in the layer
        :param activation: activation function

        WX + B
            - W shape: (output_dim, input_dim)
            - B shape: (output_dim, 1)

        N = batch_size (128)
        Layer1
            Input : (784, N)
            W     : (512, 784)
            B     : (512, 1)
            Output: (512, N)
        ...

        Layer5
            Input : (64, N)
            W     : (10, 64)
            B     : (10, 1)
            Output: (10, N)
        """

        self.weights = None
        if activation == 'relu':
            # Kaiming/He initialization
            self.weights = np.random.randn(output_dim, input_dim) * np.sqrt(2. / input_dim)
        else:
            # Xavier/Glorot initialization
            self.weights = np.random.randn(output_dim, input_dim) * np.sqrt(1. / input_dim)

        self.bias       = np.zeros((output_dim, 1))
        self.input      = None        # shape: (input_dim, batch_size)
        self.output     = None        # shape: (output_dim, batch_size)
        self.activation = activation
        self.activation_cache = None  # Derivative of activation

    def forward(self, input_data):
        # X
        input_data = input_data.astype(np.float32)
        self.input = input_data

        """
        - input_data shape: (input_dim, batch_size)
        - W          shape: (output_dim, input_dim)
        - B          shape: (output_dim, 1)
        - Z = WX + B shape: (output_dim, batch_size)
        """
        z = self.weights @ input_data + self.bias

        match self.activation:
            case 'relu':
                self.output = np.maximum(0, z)
                self.activation_cache = (z > 0).astype(z.dtype)
            case 'sigmoid':
                self.output = 1 / (1 + np.exp(-z))
                self.activation_cache = self.output * (1 - self.output)
            case 'identity':
                self.output = z
                self.activation_cache = np.ones_like(z)
            case _:
                raise ValueError(f"Unsupported activation function: {self.activation}")
        return self.output

    def backward(self, grad_output, learning_rate):
        """
        grad_output shape: (output_dim, batch_size)

        Calculate dL/dW, dL/db & dL/dX to propagate backwards
        If Z = WX + B, and out = f(z), practically: grad_z = grad_output * f'(z).

        => dW = grad_z @ X^T
        => dB = sum over batch (axis = 1) of grad_z
        => dX = W^T @ grad_z
        """
        grad_output = grad_output.astype(np.float32)
        grad_z = grad_output * self.activation_cache

        # gradient W: (output_dim, batch_size) @ (batch_size, input_dim) = (output_dim, input_dim)
        grad_weights = grad_z @ self.input.T

        # gradient B, shape: (output_dim, 1)
        grad_bias = np.sum(grad_z, axis=1, keepdims=True)

        # gradient X: W^T @ grad_z
        # W^T shape:  (input_dim, output_dim), grad_z: (output_dim, batch_size) => (input_dim, batch_size)
        grad_input = self.weights.T @ grad_z

        # Update network parameters
        self.weights -= learning_rate * grad_weights
        self.bias    -= learning_rate * grad_bias
        return grad_input  # shape: (output_dim, batch_size)

# neura_nova/ff/conv_layer.py

import numpy as np


class DenseLayer:
    # A fully connected neural network layer
    def __init__(self, input_dim, output_dim, activation='relu', learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        :param input_dim    : number of input features/neurons
        :param output_dim   : number of neurons in the layer
        :param activation   : activation function
        :param learning_rate: tasso di apprendimento per Adam
        :param beta1        : coefficiente per il primo momento     (default: 0.9)
        :param beta2        : coefficiente per il secondo momento   (default: 0.999)
        :param epsilon      : valore per evitare divisioni per zero (default: 1e-8)

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
        self.bias = np.zeros((output_dim, 1))

        self.weights = np.ascontiguousarray(self.weights, dtype=np.float32)
        self.bias    = np.ascontiguousarray(self.bias, dtype=np.float32)

        # Memoria per Adam
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_bias = np.zeros_like(self.bias)
        self.v_bias = np.zeros_like(self.bias)

        # Parametri per Adam
        self.t = 0  # step counter
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.learning_rate = learning_rate

        # Variabili di attivazione
        self.input      = None        # shape: (input_dim, batch_size)
        self.output     = None        # shape: (output_dim, batch_size)
        self.activation = activation
        self.activation_cache = None  # Derivative of activation

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

    def forward(self, input_data):
        # X
        input_data = input_data.astype(np.float32)
        self.input = input_data
        self.input = np.ascontiguousarray(input_data, dtype=np.float32)

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

    def backward(self, grad_output):
        """
        - Propagazione all'indietro con aggiornamento dei pesi usando Adam
        """
        self.t += 1  # Incrementiamo il contatore degli step
        grad_output = np.ascontiguousarray(grad_output, dtype=np.float32)

        # Calcolo del gradiente locale dL/dZ
        grad_output = grad_output.astype(np.float32)
        grad_z = grad_output * self.activation_cache

        # Gradiente in input per lo strato precedente
        grad_input = self.weights.T @ grad_z

        # gradient W: (output_dim, batch_size) @ (batch_size, input_dim) = (output_dim, input_dim)
        grad_weights = grad_z @ self.input.T

        # gradient B, shape: (output_dim, 1)
        grad_bias = np.sum(grad_z, axis=1, keepdims=True)

        # Aggiornamento dei momenti esponenziali
        self.m_weights *= self.beta1
        self.m_weights += (1 - self.beta1) * grad_weights

        self.v_weights *= self.beta2
        self.v_weights += (1 - self.beta2) * (grad_weights ** 2)

        self.m_bias *= self.beta1
        self.m_bias += (1 - self.beta1) * grad_bias

        self.v_bias *= self.beta2
        self.v_bias += (1 - self.beta2) * (grad_bias ** 2)

        # Correzione del Bias
        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t

        m_hat_weights = self.m_weights / bias_correction1
        v_hat_weights = self.v_weights / bias_correction2

        m_hat_bias = self.m_bias / bias_correction1
        v_hat_bias = self.v_bias / bias_correction2

        # Aggiornamento dei pesi con Adam
        self.weights -= self.learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)
        self.bias -= self.learning_rate * m_hat_bias / (np.sqrt(v_hat_bias) + self.epsilon)

        return grad_input

# convolutional_nn.py

import numpy as np

"""
NOTE: DO NOT RUN THIS CODE, IT IS HORRIBLE
"""

# TODO: Import MNIST dataset
# TODO: USE GPU

class Conv2D:
    """
    A simple 2D convolution layer.
    """
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        self.input_channels = input_channels
        self.output_channels = output_channels

        # Allow kernel_size to be an int or a tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights with shape: (output_channels, input_channels, kernel_h, kernel_w)
        self.weights = np.random.randn(output_channels, input_channels,
                                       self.kernel_size[0], self.kernel_size[1]) * np.sqrt(2. / (input_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.bias = np.zeros(output_channels)

    def forward(self, input):
        """
        Forward pass for convolution
        Input shape: (batch_size, input_channels, height, width)
        """
        self.input = input
        batch_size, in_channels, in_height, in_width = input.shape

        # Calculate output dimensions
        out_height = int((in_height + 2*self.padding - self.kernel_size[0]) / self.stride + 1)
        out_width = int((in_width + 2*self.padding - self.kernel_size[1]) / self.stride + 1)
        output = np.zeros((batch_size, self.output_channels, out_height, out_width))

        # Apply padding if needed
        if self.padding > 0:
            input_padded = np.pad(input, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            input_padded = input
        self.input_padded = input_padded  # Save for backward

        # Perform convolution using nested loops (TODO: this is not optimal)
        for b in range(batch_size):
            for c_out in range(self.output_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size[0]
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size[1]

                        # Extract the corresponding region
                        region = input_padded[b, :, h_start:h_end, w_start:w_end]
                        output[b, c_out, i, j] = np.sum(region * self.weights[c_out]) + self.bias[c_out]
        return output

    def backward(self, grad_output, learning_rate):
        """
        Backward pass for convolution.
        Computes gradients with respect to weights, bias, and input.
        """
        batch_size, in_channels, in_height, in_width = self.input.shape
        _, _, out_height, out_width = grad_output.shape

        grad_input_padded = np.zeros_like(self.input_padded)
        grad_weights = np.zeros_like(self.weights)
        grad_bias = np.zeros_like(self.bias)

        # Backpropagation: loop over each element of the output gradient
        for b in range(batch_size):
            for c_out in range(self.output_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size[0]
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size[1]

                        # Region from the padded input
                        region = self.input_padded[b, :, h_start:h_end, w_start:w_end]

                        # Gradients for weights and bias
                        grad_weights[c_out] += grad_output[b, c_out, i, j] * region
                        grad_bias[c_out] += grad_output[b, c_out, i, j]

                        # Gradient for the input region
                        grad_input_padded[b, :, h_start:h_end, w_start:w_end] += grad_output[b, c_out, i, j] * self.weights[c_out]

        # Remove padding
        if self.padding > 0:
            grad_input = grad_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            grad_input = grad_input_padded

        # Update parameters
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias

        return grad_input

class ReLU:
    """
    ReLU activation, operating element-wise.
    """
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, grad_output, learning_rate=0):
        grad_input = grad_output.copy()
        grad_input[self.input <= 0] = 0
        return grad_input

class Flatten:
    """
    Flattens the input while remembering the input shape for backpropagation.
    """
    def forward(self, input):
        self.input_shape = input.shape
        batch_size = input.shape[0]
        return input.reshape(batch_size, -1)

    def backward(self, grad_output, learning_rate=0):
        return grad_output.reshape(self.input_shape)

class Dense:
    """
    Fully connected layer similar to the feedforward network.
    """
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.bias = np.zeros(output_dim)

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, grad_output, learning_rate):
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0)
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        return grad_input

class MSELoss:
    """
    Mean squared error loss.
    """
    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target
        return np.mean((prediction - target)**2)

    def backward(self):
        return 2 * (self.prediction - self.target) / self.prediction.size

# TODO: Add CrossEntropy
# TODO: Add Sigmoid??

# -----------------------
# Simple CNN Model Class
# -----------------------

class CNN:
    """
    A simple CNN model combining convolution, activation, flattening, and dense layers.
    """
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, X, y, epochs, learning_rate, loss):
        for epoch in range(epochs):
            output = self.predict(X)
            l = loss.forward(output, y)
            grad = loss.backward()
            for layer in reversed(self.layers):
                grad = layer.backward(grad, learning_rate)
            if (epoch+1) % (epochs//10) == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {l:.4f}")

# Example of usage
if __name__ == "__main__":
    # Create dummy data
    # For example, treat the data as grayscale images of size 8x8
    np.random.seed(42)
    X = np.random.randn(10, 1, 8, 8)  # 10 images, 1 channel, 8x8 pixels
    # Dummy target: assume we are doing regression to output a single value for each image
    y = np.random.randn(10, 1)

    # Build a simple CNN:
    # Conv2D -> ReLU -> Flatten -> Dense
    cnn = CNN()
    cnn.add(Conv2D(input_channels=1, output_channels=4, kernel_size=3, stride=1, padding=1))
    cnn.add(ReLU())
    cnn.add(Flatten())

    # Compute flattened size: with padding=1 and stride=1, output stays 8x8, so
    # flattened size = 4 (channels) * 8 * 8 = 256
    cnn.add(Dense(256, 1))

    loss_fn = MSELoss()
    cnn.train(X, y, epochs=200, learning_rate=0.001, loss=loss_fn)

    # Test prediction with a new “image”
    X_test = np.random.randn(2, 1, 8, 8)
    predictions = cnn.predict(X_test)
    print("CNN Predictions:\n", predictions)

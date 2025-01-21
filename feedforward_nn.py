# feedforward_nn.py

import numpy as np

# TODO: Import MNIST dataset
# TODO: USE GPU

class Dense:
    """
    A fully connected neural network layer.
    """

    def __init__(self, input_dim, output_dim):
        # Random initialization for weights
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.bias = np.zeros(output_dim)

    def forward(self, input):
        """
        Forward computes WX + B.
        """
        self.input = input   # Store input for backpropagation
        return np.dot(input, self.weights) + self.bias

    def backward(self, grad_output, learning_rate):
        """
        Backward propagation:
          - Computes the gradient with respect to inputs
          - Updates parameters
        """
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0)

        # Update weights and biases
        # TODO: HERE USE ADAM OR Rprop
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        return grad_input

class ReLU:
    """
    ReLU activation function.
    """
    def forward(self, input):
        self.input = input   # Save input for backpropagation
        return np.maximum(0, input)

    def backward(self, grad_output, learning_rate=0):
        """
        Gradient passes through only where the input was positive.
        """
        grad_input = grad_output.copy()
        grad_input[self.input <= 0] = 0
        return grad_input

# TODO: Add Sigmoid

class MSELoss:
    """
    Mean Squared Error loss.
    """
    def forward(self, prediction, target):
        """
        Compute the MSE loss.
        """
        self.prediction = prediction
        self.target = target
        return np.mean((prediction - target)**2)

    def backward(self):
        """
        Gradient of MSE with respect to the predictions.
        """
        return 2 * (self.prediction - self.target) / self.prediction.size

# TODO: Add CrossEntropy

# ---------------------
# Neural Network Class
# ---------------------

class NeuralNetwork:
    """
    A simple FeedForward neural network.
    """
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer) # Add a layer to the network

    def predict(self, X):
        """
        Forward pass through all layers.
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    # TODO: What kind of training do we want to use?
    def train(self, X, y, epochs, learning_rate, loss):
        """
        Train the network:
          - Run forward pass
          - Compute loss
          - Run backward pass
          - Update parameters
        """
        for epoch in range(epochs):
            # Forward pass
            output = self.predict(X)
            l = loss.forward(output, y)

            # Backward pass: start from loss gradient
            grad = loss.backward()

            # Reverse iterate over layers for backpropagation
            for layer in reversed(self.layers):
                grad = layer.backward(grad, learning_rate)

            if (epoch+1) % (epochs//10) == 0 or epoch==0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {l:.4f}")

# Example of usage
if __name__ == "__main__":
    # Create dummy data: a simple regression where y = 2x + 3
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = 2 * X + 3

    # Build the network
    nn = NeuralNetwork()
    nn.add(Dense(1, 10))
    nn.add(ReLU())
    nn.add(Dense(10, 1))

    loss_fn = MSELoss()
    nn.train(X, y, epochs=500, learning_rate=0.01, loss=loss_fn)

    # Test prediction on new data
    X_test = np.array([[1.5], [-0.5]])
    predictions = nn.predict(X_test)
    print("Predictions:", predictions)

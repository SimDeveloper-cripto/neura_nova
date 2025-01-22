# feed_forward.py

import os
import struct
import numpy as np

# TODO: ADD ENCAPSULATION
# TODO: USE GPU
# TODO: CREATE LIBRARY

def load_mnist(path='./data'):
    def load_images(filename):
        """
        Retrieve first 16 bytes (in order):
            - Magic number
            - Number of images
            - rows
            - columns
        """
        with open(filename, 'rb') as f:
            magic_num, total, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(total, rows, cols)
        return images

    def load_labels(filename):
        """
        Retrieve first 8 bytes (in order):
            - Magic number
            - Number of images
        """
        with open(filename, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

    X_train = load_images(os.path.join(path, 'train-images-idx3-ubyte'))
    y_train = load_labels(os.path.join(path, 'train-labels-idx1-ubyte'))
    X_test  = load_images(os.path.join(path, 't10k-images-idx3-ubyte'))
    y_test  = load_labels(os.path.join(path, 't10k-labels-idx1-ubyte'))
    return (X_train, y_train), (X_test, y_test)


def one_hot_encode(y, num_classes):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot


# TODO: RENAME CLASS TO 'Layer'
class Dense:
    """
    A fully connected neural network layer.
    """

    def __init__(self, input_dim, output_dim):
        # Random initialization for weights
        self.input   = None
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        # Create bias as a row vector
        self.bias = np.zeros((1, output_dim))

    def forward(self, input_data):
        # Compute WX + B
        self.input = input_data   # Store input for backpropagation
        return np.dot(input_data, self.weights) + self.bias  # TODO: Shouldn't it be WX + B?

    def backward(self, grad_output, learning_rate):
        """
        Backward propagation:
          - Computes the gradient with respect to inputs
          - Computes the gradient for weights and biases
          - Updates parameters (SGD) # TODO: USE ADAM OR Rprop
        """
        grad_input   = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias    = np.sum(grad_output, axis=0)

        # Update
        self.weights -= learning_rate * grad_weights
        self.bias    -= learning_rate * grad_bias
        return grad_input

class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, input_data):
        self.input = input_data   # Save input for backpropagation
        return np.maximum(0, input_data)

    def backward(self, grad_output, learning_rate=0):
        """
        Gradient passes through only where the input was positive.
        """
        grad_input = grad_output.copy()
        grad_input[self.input <= 0] = 0
        return grad_input

class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, input_data):
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output

    def backward(self, grad_output, learning_rate=0):
        grad_input = grad_output * self.output * (1 - self.output)
        return grad_input


class SoftmaxCrossEntropyLoss:
    def __init__(self):
        self.probs  = None
        self.labels = None

    def forward(self, logits, labels):
        """
        Computes loss:
            - logits: raw output from the network, shape (batch_size, num_classes)
            - labels: one_hot encoded, same shape
        """
        # Softmax
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits     = np.exp(shifted_logits)
        self.probs     = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        self.labels    = labels

        # CrossEntropy
        loss = -np.sum(labels * np.log(self.probs + 1e-9)) / logits.shape[0]
        return loss

    def backward(self):
        """
        Derivative of loss respective to logits: (softmax - labels) / batch_size
        """
        batch_size = self.labels.shape[0]
        return (self.probs - self.labels) / batch_size

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
        self.layers.append(layer)

    def predict(self, input_X):
        output = input_X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    # TODO: DECIDE TRAIN STRATEGY
    def train(self, X, y, epochs, learning_rate, loss_fn, batch_size=64):
        """
        :param X: numpy array such that each row is an image (num_samples * num_features)
        :param y: target corresponding to X
        """

        num_samples = X.shape[0]
        for epoch in range(epochs):
            # Data shuffle
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

        epoch_loss = 0
        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Forward pass
            output = X_batch
            for layer in self.layers:
                output = layer.forward(output)

            loss = loss_fn.forward(output, y_batch)
            epoch_loss += loss * X_batch.shape[0]

            # Backward pass
            grad = loss_fn.backward()
            for layer in reversed(self.layers):
                grad = layer.backward(grad, learning_rate)

            epoch_loss /= num_samples
            if (epoch + 1) % max(1, epochs//10) == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    def evaluate(self, X, y):
        logits = self.predict(X)
        predictions = np.argmax(logits, axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_labels)
        return accuracy


if __name__ == "__main__":
    print("Current working directory: ", os.getcwd())

    (X_train, y_train), (X_test, y_test) = load_mnist(path='../data/MNIST/raw')

    # Pixel normalization (0..255 -> 0..1)
    X_train = X_train.astype(np.float32) / 255.0
    X_test  = X_test.astype(np.float32) / 255.0

    # Image flattening from (N, 28, 28) to (N, 784)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test  = X_test.reshape(X_test.shape[0], -1)

    num_classes = 10
    y_train_onehot = one_hot_encode(y_train, num_classes)
    y_test_onehot  = one_hot_encode(y_test, num_classes)

    # Create FeedForward NeuralNetwork
    nn = NeuralNetwork()
    nn.add(Dense(784, 128))
    nn.add(ReLU())

    nn.add(Dense(128, 64))
    nn.add(ReLU())

    nn.add(Dense(64, 10))

    loss_fn = SoftmaxCrossEntropyLoss()

    epochs = 10
    batch_size = 64
    learning_rate = 0.01

    print("(SGD) Training started...")
    nn.train(X_train, y_train_onehot, epochs, learning_rate, loss_fn, batch_size)

    test_accuracy = nn.evaluate(X_test, y_test_onehot)
    print("Test accuracy: ", test_accuracy)

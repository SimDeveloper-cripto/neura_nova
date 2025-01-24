# neura_nova/networks.py

import numpy as np
from .loss import LossFunction


class FeedForward:
    def __init__(self):
        self.layers = []
        self.loss_fn = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss_fn: LossFunction):
        self.loss_fn = loss_fn

    def predict(self, input_X):
        output = input_X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    # TODO: THIS IS MINI-BATCH, IS IT CORRECT?
    def train(self, X, y, epochs, learning_rate, loss_fn: LossFunction, batch_size=64):
        num_samples = X.shape[0]
        for epoch in range(1, epochs + 1):
            # Shuffle Data
            indices = np.random.permutation(num_samples)
            # indices = np.arange(num_samples)
            # np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0  # TODO: SHOULD IT START WITH MAX ERROR?
            for start in range(0, num_samples, batch_size):
                end     = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward
                logits     = self.predict(X_batch)
                loss       = self.loss_fn.forward(logits, y_batch)
                epoch_loss += loss * X_batch.shape[0]

                # Backward
                grad = self.loss_fn.backward()
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)

            epoch_loss /= num_samples
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")

    def evaluate(self, X, y):
        logits      = self.predict(X)
        predictions = np.argmax(logits, axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy    = np.mean(predictions == true_labels)  # TODO: THIS IS ARITHMETIC MEAN
        return accuracy


class Convolutional:
    pass
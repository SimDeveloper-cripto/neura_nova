# neura_nova/networks.py

import numpy as np
from .loss import LossFunction


class FeedForward:
    def __init__(self, loss_fn: LossFunction):
        self.layers    = []
        self.loss_fn   = loss_fn
        self.__history = {
            "loss": [],
            "accuracy": []
        }

    def add_layer(self, layer):
        self.layers.append(layer)

    def getHistory(self):
        return self.__history

    def predict(self, input_X):
        """
        input_X shape: (input_dim, batch_size)
        output  shape: (output_dim, batch_size)
        """
        output = input_X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, X, y, epochs, learning_rate, batch_size=64):
        """
        X shape: (input_dim, N)
        y shape: (num_classes, N)
        """
        num_samples = X.shape[1]

        for epoch in range(1, epochs + 1):
            # Shuffle
            indices = np.random.permutation(num_samples)
            X_shuffled = X[:, indices]  # (input_dim, N)
            y_shuffled = y[:, indices]  # (num_classes, N)

            epoch_loss = 0.0
            for start in range(0, num_samples, batch_size):
                end     = start + batch_size
                X_batch = X_shuffled[:, start:end]  # (input_dim, batch_size)
                y_batch = y_shuffled[:, start:end]  # (num_classes, batch_size)

                # Forward
                logits     = self.predict(X_batch)  # (num_classes, batch_size)
                loss       = self.loss_fn.forward(logits, y_batch)
                epoch_loss += loss * X_batch.shape[1]

                # Backward
                grad = self.loss_fn.backward()      # (num_classes, batch_size)
                for layer in reversed(self.layers):
                    # TODO: ADATTA QUESTO CODICE PER UTILIZZARE GLI OTTIMIZZATORI
                    grad = layer.backward(grad, learning_rate)

            epoch_loss /= num_samples
            epoch_accuracy = self.evaluate(X_shuffled, y_shuffled)
            print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")

            self.__history["loss"].append(epoch_loss)
            self.__history["accuracy"].append(epoch_accuracy)

    def evaluate(self, X, y):
        """
        X shape: (input_dim, N)
        y shape: (num_classes, N)
        """
        logits      = self.predict(X)
        predictions = np.argmax(logits, axis=0)  # shape: (N,)
        true_labels = np.argmax(y, axis=0)       # shape: (N,)
        accuracy    = np.mean(predictions == true_labels)  # TODO: THIS IS ARITHMETIC MEAN, VA BENE?
        return accuracy


class Convolutional:
    pass

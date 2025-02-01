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
                logits     = self.predict(X_batch)
                loss       = self.loss_fn.forward(logits, y_batch)
                epoch_loss += loss * X_batch.shape[1]

                # Backward
                grad = self.loss_fn.backward()
                for layer in reversed(self.layers):
                    grad = layer.backward(grad)

            epoch_loss /= num_samples
            epoch_accuracy = self.arithmetic_mean_accuracy(X_shuffled, y_shuffled)
            print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")

            self.__history["loss"].append(epoch_loss)
            self.__history["accuracy"].append(epoch_accuracy)

    def arithmetic_mean_accuracy(self, X, y):
        """
        X shape  : (input_dim, N)
        y shape  : (num_classes, N)
        precision: boolean to specify the usage of PRECISION algorithm
        """

        """
        - [GOOD] Arithmetic mean
            - Accuracy = (# of correct predictions) / (# of predictions in total)
        MNIST contains approximately equal numbers of samples for each of the 10 classes.
        In this context, accuracy is an effective metric because it is not affected by class imbalances.
        """
        logits      = self.predict(X)
        predictions = np.argmax(logits, axis=0)  # shape: (N,)
        true_labels = np.argmax(y, axis=0)       # shape: (N,)
        accuracy    = np.mean(predictions == true_labels)
        return accuracy

"""
    def precision_accuracy(self, X, y, message):
        # Precision = TP / (TP + FP)
        # Precision Macro = 1/C * sum(i = 1 to C) of Precision_i
        # Precision Micro = sum(i = 1 to C) of TP_i / sum(i = 1 to C) of TP_i + FP_i
        num_classes = 10
        predictions, true_labels = self.__evaluate(X, y)

        precision_per_class = []
        TP_total = 0
        FP_total = 0

        for cls in range(num_classes):
            TP = np.sum((predictions == cls) & (true_labels == cls))
            FP = np.sum((predictions == cls) & (true_labels != cls))
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            precision_per_class.append(precision)
            TP_total += TP
            FP_total += FP

        precision_macro = np.mean(precision_per_class)
        precision_micro = TP_total / (TP_total + FP_total) if (TP_total + FP_total) > 0 else 0.0

        print(message)
        for cls in range(num_classes):
            print(f"Precision Class [{cls}]: {precision_per_class[cls] * 100:.2f}%")
        print(f"Precision Macro: {precision_macro * 100:.2f}%")
        print(f"Precision Micro: {precision_micro * 100:.2f}%")

        return {
            "precision_per_class": precision_per_class,
            "precision_macro": precision_macro,
            "precision_micro": precision_micro
        }
"""


class Convolutional:
    # TODO: DA IMPLEMENTARE
    pass

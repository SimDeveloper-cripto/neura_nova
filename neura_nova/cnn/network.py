# neura_nova/cnn/network.py

import numpy as np
from ..loss import LossFunction


class Network:
    def train(self, X, y, epochs, learning_rate, batch_size=128):
        raise NotImplementedError

class Convolutional(Network):
    def __init__(self, loss_fn: LossFunction):
        self.conv_layers  = []
        self.pool_layers  = []
        self.loss_fn      = loss_fn
        self.__history    = {
            "loss": [],
            "accuracy": []
        }

        self.fc_layer = None

    def add_conv_layer(self, layer):
        self.conv_layers.append(layer)

    def add_pool_layer(self, layer):
        self.pool_layers.append(layer)

    def getHistory(self):
        return self.__history

    def predict(self, input_X):
        output = input_X

        for conv in self.conv_layers:
            output = conv.forward(output)

        for pool in self.pool_layers:
            output = pool.forward(output)

        batch_size = output.shape[0]

        # Flattening: from (N, C, H, W) to (N, C * H * W)
        flattened = output.reshape(batch_size, -1).T

        if self.fc_layer is None:
            raise ValueError("Fully-Connected layer is not initialized.")

        Z = self.fc_layer.forward(flattened)
        return Z

    def train(self, X, y, epochs, learning_rate, batch_size=128):
        num_samples = X.shape[0]

        for epoch in range(1, epochs + 1):
            # Shuffle
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            for start in range(0, num_samples, batch_size):
                end     = start + batch_size
                X_batch = X_shuffled[start:end]  # (batch_size, channels, H, W)
                y_batch = y_shuffled[start:end]  # (batch_size, channels, H, W)

                # Forward
                out = X_batch
                for conv in self.conv_layers:
                    out = conv.forward(out)
                for pool in self.pool_layers:
                    out = pool.forward(out)

                out_shape          = out.shape  # Save shape for backward
                current_batch_size = out.shape[0]
                flattened          = out.reshape(current_batch_size, -1).T

                Z = self.fc_layer.forward(flattened)

                loss = self.loss_fn.forward(Z, y_batch.T)
                epoch_loss += loss * current_batch_size

                # Backward
                # Backprop from the fully-connected layer
                grad_loss = self.loss_fn.backward()  # grad_loss: (10, batch_size)
                grad_fc   = self.fc_layer.backward(grad_loss)
                grad      = grad_fc.T.reshape(out_shape)

                for pool in reversed(self.pool_layers):
                    grad = pool.backward(grad)
                for conv in reversed(self.conv_layers):
                    grad = conv.backward(grad)

            epoch_loss /= num_samples
            epoch_accuracy = self.arithmetic_mean_accuracy(X, y)
            print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

            self.__history["loss"].append(epoch_loss)
            self.__history["accuracy"].append(epoch_accuracy)

    def arithmetic_mean_accuracy(self, X, y):
        logits = self.predict(X)
        predictions = np.argmax(logits, axis=0)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_labels)
        return accuracy

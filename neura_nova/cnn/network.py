# neura_nova/cnn/network.py

import numpy as np
from ..loss import LossFunction


class Network:
    def train(self, X, y, epochs, X_val, y_val, learning_rate, batch_size=128):
        raise NotImplementedError

class Convolutional(Network):
    def __init__(self, loss_fn: LossFunction):
        self.conv_layers  = []
        self.pool_layers  = []
        self.fc_layers    = []
        self.loss_fn      = loss_fn
        self.__train_history    = {
            "train_loss": [],
            "train_accuracy": []
        }
        self.__validation_history    = {
            "validation_loss": [],
            "validation_accuracy": []
        }

    def add_conv_layer(self, layer):
        self.conv_layers.append(layer)

    def add_pool_layer(self, layer):
        self.pool_layers.append(layer)

    def getTrainHistory(self):
        return self.__train_history

    def getValidationHistory(self):
        return self.__validation_history

    def predict(self, input_X):
        output = input_X
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            output = conv.forward(output)
            output = pool.forward(output)

        batch_size = output.shape[0]

        # Flattening: from (N, C, H, W) to (N, C * H * W)
        flattened = output.reshape(batch_size, -1).T

        if self.fc_layers is None:
            raise ValueError("Fully-Connected layers are not initialized.")

        Z = flattened
        for fc in self.fc_layers:
            Z = fc.forward(Z)
        return Z

    def train(self, X, y, epochs, X_val, y_val, learning_rate, batch_size=128, patience=6):
        num_samples = X.shape[0]

        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None

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
                for conv, pool in zip(self.conv_layers, self.pool_layers):
                    out = conv.forward(out)
                    out = pool.forward(out)

                out_shape          = out.shape  # Save shape for backward
                current_batch_size = out.shape[0]
                flattened          = out.reshape(current_batch_size, -1).T

                Z = flattened
                for fc in self.fc_layers:
                    Z = fc.forward(Z)

                loss = self.loss_fn.forward(Z, y_batch.T)
                epoch_loss += loss * current_batch_size

                # Backward
                # Backprop from the fully-connected layer
                grad = self.loss_fn.backward()  # grad: (10, batch_size)
                for fc in reversed(self.fc_layers):
                    grad = fc.backward(grad)
                grad = grad.T.reshape(out_shape)

                for conv, pool in reversed(list(zip(self.conv_layers, self.pool_layers))):
                    grad = pool.backward(grad)
                    grad = conv.backward(grad)

            epoch_loss /= num_samples
            epoch_accuracy = self.arithmetic_mean_accuracy(X, y)

            val_logits = self.predict(X_val)
            val_loss = self.loss_fn.forward(val_logits, y_val.T)
            val_accuracy = self.arithmetic_mean_accuracy(X_val, y_val)

            print(f"epoch {epoch}/{epochs}: train_loss: {epoch_loss:.4f},"
                  f"train_accuracy: {epoch_accuracy:.4f}, val_loss: {val_loss:.4f}, val_accuracy: {val_accuracy:.4f}")

            self.__train_history["train_loss"].append(epoch_loss)
            self.__train_history["train_accuracy"].append(epoch_accuracy)
            self.__validation_history["validation_loss"].append(epoch_loss)
            self.__validation_history["validation_accuracy"].append(epoch_accuracy)

            # EARLY STOPPING
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = [layer.get_weights() for layer in self.conv_layers]
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if best_weights:
            for layer, best_weight in zip(self.conv_layers, best_weights):
                layer.set_weights(best_weight)

    def arithmetic_mean_accuracy(self, X, y):
        logits = self.predict(X)
        predictions = np.argmax(logits, axis=0)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_labels)
        return accuracy

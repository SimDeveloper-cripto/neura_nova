# neura_nova/ff/network.py

import numpy as np
from ..loss import LossFunction


class Network:
    def predict(self, input_X):
        raise NotImplementedError

    def train(self, X, y, epochs, X_val, y_val, batch_size):
        raise NotImplementedError

    def getAccuracy(self, X_test, y_test_onehot, dataset_size):
        logits      = self.predict(X_test)
        predictions = np.argmax(logits, axis=0)
        true_labels = np.argmax(y_test_onehot, axis=0)
        correct     = np.sum(predictions == true_labels)
        return correct / dataset_size

    def arithmetic_mean(self, X, y):
        logits = self.predict(X)
        predictions = np.argmax(logits, axis=0)
        true_labels = np.argmax(y, axis=0)
        accuracy = np.mean(predictions == true_labels)
        return accuracy

class FeedForward(Network):
    def __init__(self, loss_fn: LossFunction):
        self.layers    = []
        self.loss_fn   = loss_fn

        self.__train_history = {
            "train_loss": [],
            "train_accuracy": []
        }
        self.__validation_history = {
            "validation_loss": [],
            "validation_accuracy": []
        }

    def add_layer(self, layer):
        self.layers.append(layer)

    def predict(self, input_X):
        output = input_X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def getTrainHistory(self):
        return self.__train_history

    def getValidationHistory(self):
        return self.__validation_history

    def train(self, X, y, epochs, X_val, y_val, batch_size, stopping_criterion=10):
        num_samples = X.shape[1]

        best_val_loss = float('inf')
        stopping_counter = 0
        best_weights = None

        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(num_samples)
            X_shuffled = X[:, indices]  # (input_dim, N)
            y_shuffled = y[:, indices]  # (num_classes, N)

            epoch_loss = 0.0
            for start in range(0, num_samples, batch_size):
                end     = start + batch_size
                X_batch = X_shuffled[:, start:end]  # (input_dim, batch_size)
                y_batch = y_shuffled[:, start:end]  # (num_classes, batch_size)

                logits     = self.predict(X_batch)
                loss       = self.loss_fn.forward(logits, y_batch)
                epoch_loss += loss * X_batch.shape[1]

                grad = self.loss_fn.backward()
                for layer in reversed(self.layers):
                    grad = layer.backward(grad)

            epoch_loss /= num_samples
            val_logits = self.predict(X_val)
            val_loss   = self.loss_fn.forward(val_logits, y_val)

            print(f"epoch {epoch}/{epochs}, train_loss: {epoch_loss:.4f} val_loss: {val_loss:.4f}")

            epoch_accuracy = self.arithmetic_mean(X_shuffled, y_shuffled)
            val_accuracy   = self.arithmetic_mean(X_val, y_val)

            self.__train_history["train_loss"].append(epoch_loss)
            self.__train_history["train_accuracy"].append(epoch_accuracy)
            self.__validation_history["validation_loss"].append(val_loss)
            self.__validation_history["validation_accuracy"].append(val_accuracy)

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                stopping_counter = 0
                best_weights     = [layer.get_weights() for layer in self.layers]  # Salvataggio pesi migliori
            else:
                stopping_counter += 1

            if stopping_counter >= stopping_criterion:
                print(f"Early stopping at epoch {epoch}")
                break

        if best_weights:
            for layer, best_weight in zip(self.layers, best_weights):
                layer.set_weights(best_weight)
# neura_nova/api.py

import asyncio
import numpy as np

from .layers import DenseLayer
from .networks import FeedForward
from .loss import SoftmaxCrossEntropyLoss

from .data import load_mnist
from .utils import visualize_predictions, plot_metrics


def one_hot_encode(y, num_classes):
    # return np.eye(num_classes)[y]
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

# TODO: USE GPU
# TODO: USE `cProfile` FOR BOTTLENECKS
# TODO: SAVE MODEL WITH DIFFERENT TRAINING STRATEGIES FOR BOTH TYPES
# TODO: THE LEARNING IS TOO SLOW
# TODO: WHERE DOES FEATURE EXTRACTION TAKE PLACE?
# TODO: HOW MANY LAYERS AND HOW MANY NEURONS PER LAYER ARE THERE?
# TODO: 'sigmoid' MAKES LEARNING WORSE
# TODO: is there a bound to the loss function? Or can I normalize it?
# TODO: SHOW TEST ACCURACY PLOT
# TODO: MAKE SURE `visualize_predictions` DOES NOT INTRODUCE USELESS OVERHEAD

# TODO: LA FUNZIONE SI PUO' SISTEMARE
def build_and_train_model_with_sgd():
    print("[INFO] LOADING MNIST DATASET")
    (X_train, y_train), (X_test, y_test) = load_mnist()

    print(f"[INFO] TRAIN SET SIZE: {X_train.shape[0]}")
    print(f"[INFO] TEST SET SIZE: {X_test.shape[0]}")

    # Pre-Processing (normalization)
    X_train = X_train.astype(np.float32) / 255.0
    X_test  = X_test.astype(np.float32) / 255.0

    # Flatten images
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test  = X_test.reshape(X_test.shape[0], -1)

    num_classes = 10
    y_train_onehot = one_hot_encode(y_train, num_classes)
    y_test_onehot  = one_hot_encode(y_test, num_classes)

    labels = np.arange(num_classes)
    one_hot_matrix = one_hot_encode(labels, num_classes)
    print("")
    print(one_hot_matrix)

    # Create Feed-Forward Neural Network
    nn = FeedForward(SoftmaxCrossEntropyLoss())
    # TODO: FIX NETWORK STRUCTURE
    nn.add_layer(DenseLayer(784, 128, activation='relu'))
    nn.add_layer(DenseLayer(128, 64, activation='relu'))

    # TODO: MIGHT WANT TO USE A DIFFERENT ACTIVATION HERE
    nn.add_layer(DenseLayer(64, 10, activation='identity'))

    epochs = 20
    batch_size = 64
    learning_rate = 0.01

    print("\n[INFO] STARTED TRAINING WITH `SGD UPDATE RULE`:")
    nn.train(X_train, y_train_onehot, epochs, learning_rate, batch_size)

    # Evaluate learning accuracy
    train_accuracy = nn.evaluate(X_train, y_train_onehot)
    print("\n[INFO] TRAIN ACCURACY: {:.2f}%".format(train_accuracy * 100))

    # Plot the graph
    asyncio.run(plot_metrics("TRAIN: LOSS FUNCTION", nn.getHistory(), metric_names=["loss", "accuracy"]))

    # Evaluate test accuracy
    test_accuracy = nn.evaluate(X_test, y_test_onehot)
    print("[INFO] TEST ACCURACY: {:.2f}%".format(test_accuracy * 100))

    # Visualize predictions
    asyncio.run(visualize_predictions(nn, X_test, y_test_onehot))
    print("[STATUS] PROGRAM ENDED")

def build_and_train_model_with_adam():
    pass

def build_and_train_model_with_rprop():
    pass

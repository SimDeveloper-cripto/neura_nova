# neura_nova/api.py

import asyncio
import numpy as np
from .layers import DenseLayer
from .networks import FeedForward
from .loss import SoftmaxCrossEntropyLoss

from .data import load_mnist
from .utils import visualize_predictions


def one_hot_encode(y, num_classes):
    # return np.eye(num_classes)[y]
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

# TODO: USE GPU
# TODO: SAVE MODEL WITH DIFFERENT TRAINING STRATEGIES FOR BOTH TYPES
# TODO: THE LEARNING IS TOO SLOW
# TODO: WHERE DOES FEATURE EXTRACTION TAKE PLACE?
# TODO: HOW MANY LAYERS AND HOW MANY NEURONS PER LAYER ARE THERE?


def build_and_train_model():
    # Builds, Trains, and Evaluates a neural network based on the MNIST dataset
    print("[INFO] LOADING MNIST DATASET")
    (X_train, y_train), (X_test, y_test) = load_mnist()

    # Pre-Processing
    X_train = X_train.astype(np.float32) / 255.0
    X_test  = X_test.astype(np.float32) / 255.0

    # Flatten images
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test  = X_test.reshape(X_test.shape[0], -1)

    num_classes = 10
    y_train_onehot = one_hot_encode(y_train, num_classes)
    y_test_onehot  = one_hot_encode(y_test, num_classes)

    # Create Feed-Forward Neural Network
    nn = FeedForward()
    nn.add_layer(DenseLayer(784, 128, activation='relu'))
    nn.add_layer(DenseLayer(128, 64, activation='relu'))

    # TODO: MIGHT WANT TO USE A DIFFERENT ACTIVATION HERE
    nn.add_layer(DenseLayer(64, 10, activation='identity'))

    loss_fn = SoftmaxCrossEntropyLoss()
    nn.set_loss(loss_fn)

    epochs = 20
    batch_size = 64
    learning_rate = 0.01

    print("[INFO] STARTED TRAINING WITH `SGD UPDATE RULE`\n")
    nn.train(X_train, y_train_onehot, epochs, learning_rate, loss_fn, batch_size)

    test_accuracy = nn.evaluate(X_test, y_test_onehot)
    print("\n[INFO] TEST ACCURACY: {:.2f}%".format(test_accuracy * 100))

    # TODO: MAKE SURE THIS DOES NOT INTRODUCE USELESS OVERHEAD
    asyncio.run(visualize_predictions(nn, X_test, y_test_onehot))

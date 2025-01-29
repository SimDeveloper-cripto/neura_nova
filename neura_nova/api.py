# neura_nova/api.py

import asyncio
import numpy as np

from .layers import DenseLayer
from .networks import FeedForward
from .loss import SoftmaxCrossEntropyLoss

from .data import load_mnist
from .utils import visualize_predictions, plot_metrics

# TODO: USE GPU AND SPEED-UP THE PROGRAM
# TODO: USE `cProfile` FOR BOTTLENECKS
# TODO: SAVE MODEL WITH DIFFERENT TRAINING STRATEGIES FOR BOTH TYPES
# TODO: WHERE DOES FEATURE EXTRACTION TAKE PLACE?
# TODO: HOW MANY LAYERS AND HOW MANY NEURONS PER LAYER ARE THERE?
# TODO: is there a bound to the loss function? Or can I normalize it?

def load_and_preprocess_data():
    (X_train, y_train), (X_test, y_test) = load_mnist()

    # print(f"[INFO] TRAIN SET SIZE: {X_train.shape[0]}")
    # print(f"[INFO] TEST SET SIZE: {X_test.shape[0]}")

    # Pre-Processing (normalization)
    X_train = X_train.astype(np.float32) / 255.0
    X_test  = X_test.astype(np.float32) / 255.0

    # Flatten images
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test  = X_test.reshape(X_test.shape[0], -1)

    num_classes = 10
    y_train_onehot = one_hot_encode(y_train, num_classes)
    y_test_onehot  = one_hot_encode(y_test, num_classes)

    return X_train, y_train_onehot, X_test, y_test_onehot

def one_hot_encode(y, num_classes):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

def build_ff_model(loss_fun=SoftmaxCrossEntropyLoss()):
    nn = FeedForward(loss_fun)
    nn.add_layer(DenseLayer(784, 512, activation='relu'))
    nn.add_layer(DenseLayer(512, 256, activation='relu'))
    nn.add_layer(DenseLayer(256, 128, activation='relu'))
    nn.add_layer(DenseLayer(128, 64,  activation='relu'))
    nn.add_layer(DenseLayer(64,  10,  activation='identity'))

    # Layer 1: input_dim = 784, output_dim = 512, Weight_M = 784x512, Bias_M = 1x512
    # Layer 2: input_dim = 512, output_dim = 256, Weight_M = 512x256, Bias_M = 1x256
    # Layer 3: input_dim = 256, output_dim = 128, Weight_M = 256x128, Bias_M = 1x128
    # Layer 4: input_dim = 128, output_dim = 64,  Weight_M = 128x64,  Bias_M = 1x64
    # Layer 5: input_dim = 64,  output_dim = 10,  Weight_M = 64x10,   Bias_M = 1x10

    return nn

def evaluate_model(model, X, y, dataset="TRAIN"):
    accuracy = model.evaluate(X, y)
    print(f"\n[INFO] {dataset} ACCURACY: {accuracy * 100:.2f}%")
    return accuracy

def build_and_train_model():
    X_train, y_train_onehot, X_test, y_test_onehot = load_and_preprocess_data()
    """
        labels = np.arange(10)
        one_hot_matrix = one_hot_encode(labels, 10)
        print("")
        print(one_hot_matrix)
    """
    nn = build_ff_model()

    epochs = 20
    batch_size = 64
    learning_rate = 0.01

    # SGD update rule
    nn.train(X_train, y_train_onehot, epochs, learning_rate, batch_size)

    # Evaluate learning and test accuracy
    train_accuracy = nn.evaluate(X_train, y_train_onehot)
    test_accuracy  = nn.evaluate(X_test, y_test_onehot)

    print("\n[INFO] TRAIN ACCURACY: {:.2f}%".format(train_accuracy * 100))
    print("[INFO] TEST ACCURACY: {:.2f}%".format(test_accuracy * 100))

    asyncio.run(plot_metrics("TRAIN: LOSS FUNCTION", nn.getHistory(), metric_names=["loss", "accuracy"]))
    asyncio.run(visualize_predictions(nn, X_test, y_test_onehot))

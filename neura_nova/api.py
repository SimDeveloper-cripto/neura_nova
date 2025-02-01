# neura_nova/api.py

import numpy as np

from .ff_layer import DenseLayer
from .networks import FeedForward
from .loss import SoftmaxCrossEntropy

from .data import load_mnist_ff
from .graphic_utils import visualize_predictions, plot_metrics

# TODO: USA GPU
# TODO: MEDIA ARITMETICA OPPURE PRECISION PER ACCURACY

def one_hot_encode(y, num_classes):
    one_hot = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    one_hot[np.arange(y.shape[0]), y] = 1.0
    return one_hot

def load_and_preprocess_data_for_ff():
    (X_train, y_train), (X_test, y_test) = load_mnist_ff()

    # print(f"[INFO] TRAIN SET SIZE: {X_train.shape[0]}")
    # print(f"[INFO] TEST SET SIZE:  {X_test.shape[0]}")

    # Pre-Processing (normalization)
    X_train = X_train.astype(np.float32) / 255.0
    X_test  = X_test.astype(np.float32) / 255.0

    # Flatten images
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test  = X_test.reshape(X_test.shape[0], -1)

    num_classes = 10
    y_train_onehot = one_hot_encode(y_train, num_classes)  # shape: (N, 10)
    y_test_onehot  = one_hot_encode(y_test, num_classes)

    # Use WX + B convention
    X_train        = X_train.T         # shape: (784, N_train)
    X_test         = X_test.T          # shape: (784, N_test)
    y_train_onehot = y_train_onehot.T  # shape: (10,  N_train)
    y_test_onehot  = y_test_onehot.T   # shape: (10,  N_test)
    return X_train, y_train_onehot, X_test, y_test_onehot

def build_ff_model(loss_fun=SoftmaxCrossEntropy()):
    """
    - input_dim     = 784
    - output_dim    = 10
    - weights shape = (output_dim, input_dim)
    """
    nn = FeedForward(loss_fun)
    # See ff_layer.py __init__() and forward()
    nn.add_layer(DenseLayer(784, 512, activation='relu'))       # 512 neurons, W: (512, 784)
    nn.add_layer(DenseLayer(512, 256, activation='relu'))       # 256 neurons, W: (256, 512)
    nn.add_layer(DenseLayer(256, 128, activation='relu'))       # 128 neurons, W: (128, 256)
    nn.add_layer(DenseLayer(128, 64,  activation='relu'))       # 64  neurons, W: (64,  128)
    nn.add_layer(DenseLayer(64,  10,  activation='identity'))   # 10  neurons, W: (10,  64)
    return nn

def build_and_train_ff_model():
    X_train, y_train_onehot, X_test, y_test_onehot = load_and_preprocess_data_for_ff()
    nn = build_ff_model()

    epochs = 15
    batch_size = 128
    learning_rate = 0.001

    # Using ADAM update rule
    nn.train(X_train, y_train_onehot, epochs, learning_rate, batch_size)

    # Evaluate learning and test accuracy
    train_accuracy = nn.arithmetic_mean_accuracy(X_train, y_train_onehot)
    test_accuracy  = nn.arithmetic_mean_accuracy(X_test, y_test_onehot)

    print("\n[INFO] TRAIN ARITHMETIC_MEAN_ACCURACY: {:.2f}%".format(train_accuracy * 100))
    print("[INFO] TEST ARITHMETIC_MEAN_ACCURACY: {:.2f}%".format(test_accuracy * 100))

    # plot_metrics("TRAIN: LOSS FUNCTION", nn.getHistory(), metric_names=["loss", "accuracy"])
    # visualize_predictions(nn, X_test, y_test_onehot)

"""
    train_accuracy = nn.precision_accuracy(X_train, y_train_onehot, "----- TRAIN")
    print("\n[INFO] TRAIN PRECISION_ACCURACY: ", train_accuracy)

    test_accuracy = nn.precision_accuracy(X_test, y_test_onehot, "----- TEST")
    print("\n[INFO] TEST PRECISION_ACCURACY: ", test_accuracy)
"""

def build_cnn_model(loss_fun=SoftmaxCrossEntropy()):
    pass

def build_and_train_cnn_model():
    pass
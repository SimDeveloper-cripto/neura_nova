import math
import array
import random
import numpy as np

from ..data import load_mnist

from .layer import DenseLayer
from .network import FeedForward
from ..loss import SoftmaxCrossEntropy


def closest_power_of_2(n):
    lower = 2 ** math.floor(math.log2(n))
    upper = 2 ** math.ceil(math.log2(n))
    return lower if (n - lower) < (upper - n) else upper

def create_neurons(input_dim, output_dim, hidden_layers):
    neurons = array.array('i', [input_dim])
    i = 1
    while i < hidden_layers:
        last_layer = neurons[-1]
        while True:
            random_number = random.randint(last_layer // 2, last_layer)
            neurons_in_layer = closest_power_of_2(random_number)
            if neurons_in_layer != neurons[-1]:
                break

        neurons.append(neurons_in_layer)
        i += 1
    neurons.append(output_dim)
    return neurons

def one_hot_encode(y, num_classes):
    one_hot = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    one_hot[np.arange(y.shape[0]), y] = 1.0
    return one_hot

def load_and_preprocess_data_for_ff():
    (X_train, y_train), (X_test, y_test) = load_mnist()

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
    - weights shape = (output_dim, input_dim) = (# neurons, # features)
    """
    nn = FeedForward(loss_fun)  # See conv_layer.py __init__() and forward()

    # Qui creo l'array di neuroni
    layers = create_neurons(784, 10, 5)

    # layers[0]: input vector
    # layers[5]: output neurons
    nn.add_layer(DenseLayer(layers[0], layers[1], activation='relu'))      # layers[1] neurons,  W: (layers[1], 784)
    nn.add_layer(DenseLayer(layers[1], layers[2], activation='relu'))      # layers[2] neurons,  W: (layers[2], layers[1])
    nn.add_layer(DenseLayer(layers[2], layers[3], activation='relu'))      # layers[3] neurons,  W: (layers[3], layers[2])
    nn.add_layer(DenseLayer(layers[3], layers[4], activation='relu'))      # layers[4]  neurons, W: (layers[4], layers[3])
    nn.add_layer(DenseLayer(layers[4], layers[5], activation='identity'))  # layers[5]  neurons, W: (layers[5], layers[4])

    '''
    nn.add_layer(DenseLayer(784, 512, activation='relu'))       # 512 neurons, W: (512, 784)
    nn.add_layer(DenseLayer(512, 256, activation='relu'))       # 256 neurons, W: (256, 512)
    nn.add_layer(DenseLayer(256, 128, activation='relu'))       # 128 neurons, W: (128, 256)
    nn.add_layer(DenseLayer(128, 64,  activation='relu'))       # 64  neurons, W: (64,  128)
    nn.add_layer(DenseLayer(64,  10,  activation='identity'))   # 10  neurons, W: (10,  64)    
    '''
    return nn

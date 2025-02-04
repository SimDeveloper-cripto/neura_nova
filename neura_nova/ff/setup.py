import os
import csv
import math
import array
import random
import numpy as np

from ..data import load_mnist

from .layer import DenseLayer
from .network import FeedForward
from ..loss import SoftmaxCrossEntropy

def save_to_csv(results, filename='results/ff/results.csv'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    headers = ["train_accuracy", "test_accuracy", "epochs", "batch_size"]

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(headers)

        for result in results:
            writer.writerow([result['train_accuracy'], result['test_accuracy'], result['epochs'], result['batch_size']])

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

def load_and_preprocess_data_for_ff(train_limit, test_limit):
    (X_train, y_train), (X_test, y_test) = load_mnist(train_limit, test_limit)

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

def build_and_train_ff_model_with_config(config, loss_fun=SoftmaxCrossEntropy()):
    """
    - input_dim     = 784
    - output_dim    = 10
    - weights shape = (output_dim, input_dim) = (# neurons, # features)
    """
    X_train, y_train_onehot, X_test, y_test_onehot = load_and_preprocess_data_for_ff(config['train_dimension'], config['test_dimension'])
    nn = FeedForward(loss_fun)  # See conv_layer.py __init__() and forward()

    for layer_config in config['layers']:
        nn.add_layer(DenseLayer(input_dim=784 if len(nn.layers) == 0 else nn.layers[-1].weights.shape[0],
                                output_dim=layer_config['neurons'],
                                activation=layer_config['activation']))

    epochs     = config['epochs']
    batch_size = config['batch_size']
    nn.train(X_train, y_train_onehot, epochs, 0.001, batch_size)

    train_accuracy = nn.arithmetic_mean_accuracy(X_train, y_train_onehot)
    test_accuracy  = nn.arithmetic_mean_accuracy(X_test, y_test_onehot)

    # Read ffconfig.json
    result = {
        'train_accuracy': "{:.2f}".format(train_accuracy * 100),
        'test_accuracy': "{:.2f}".format(test_accuracy * 100),
        'epochs': epochs,
        'batch_size': batch_size,
    }
    return result

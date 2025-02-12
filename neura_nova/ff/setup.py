import os
import json
import math
import array
import random
import numpy as np

from ..data import load_mnist

from .layer import DenseLayer
from .network import FeedForward
from ..loss import SoftmaxCrossEntropy

def save_ff_to_json(results, filename='results/ff/results.json'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        json.dump([results], file, indent=4)

    print("results/ff/results.json has been updated!")

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

def load_and_preprocess_data_for_ff(train_limit, test_limit, validation_limit):
    (X_train, y_train), (X_test, y_test) = load_mnist(train_limit, test_limit)

    # Pre-Processing (normalization)
    X_train = X_train.astype(np.float32) / 255.0
    X_test  = X_test.astype(np.float32) / 255.0

    # Flatten images
    X_train = X_train.reshape(X_train.shape[0], -1)  # shape: (N_train, 784)
    X_test  = X_test.reshape(X_test.shape[0], -1)    # shape: (N_train, 784)

    num_classes = 10
    y_train_onehot = one_hot_encode(y_train, num_classes)  # shape: (N, 10)
    y_test_onehot  = one_hot_encode(y_test, num_classes)

    # % train_set + % validation_set
    X_train_final = X_train[:validation_limit]
    y_train_final = y_train_onehot[:validation_limit]
    X_val         = X_train[validation_limit:]
    y_val         = y_train_onehot[validation_limit:]

    # Use WX + B convention
    # Training
    X_train_final = X_train_final.T  # shape: (784,  N_train_final)
    y_train_final = y_train_final.T  # shape: (10,   N_train_final)

    # Validation
    X_val         = X_val.T          # shape: (784,  N_val)
    y_val         = y_val.T          # shape: (10,   N_val)

    # Test
    X_test        = X_test.T         # shape: (7_84, N_test)
    y_test_onehot = y_test_onehot.T  # shape: (10,   N_test)

    return X_train_final, y_train_final, X_val, y_val, X_test, y_test_onehot

def build_and_train_ff_model_with_config(config, loss_fun=SoftmaxCrossEntropy()):
    """
    - input_dim     = 784
    - output_dim    = 10
    - weights shape = (output_dim, input_dim) = (# neurons, # features)
    """
    X_train, y_train_onehot, X_val, y_val, X_test, y_test_onehot = load_and_preprocess_data_for_ff(config['train_dimension'], config['test_dimension'], config['validation_dimension'])
    nn = FeedForward(loss_fun)  # See conv_layer.py __init__() and forward()

    for layer_config in config['layers']:
        nn.add_layer(DenseLayer(input_dim=784 if len(nn.layers) == 0 else nn.layers[-1].weights.shape[0],
                                output_dim=layer_config['neurons'],
                                activation=layer_config['activation']))

    epochs     = config['epochs']
    batch_size = config['batch_size']
    lr         = 0.001
    nn.train(X_train, y_train_onehot, epochs, X_val, y_val, lr, batch_size)

    # Sulla base dei pesi migliori
    test_accuracy = 0  # TODO: nn.getAccuracy(X_test, y_test_onehot, config['test_dimension'])

    # Read config/ffconfig.json
    result = {
        'layers': config['layers'],
        'train_dimension': config['train_dimension'],
        'test_dimension': config['test_dimension'],
        'epochs': epochs,
        'batch_size': batch_size,
        'test_accuracy': "{:.2f}".format(test_accuracy * 100)
    }
    return result

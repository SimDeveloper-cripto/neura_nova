import os
import json
import numpy as np

from .network import Convolutional

from ..data import load_mnist
from ..ff.layer import DenseLayer
from ..loss import SoftmaxCrossEntropy

from .conv_layer import ConvLayer
from .pool_layer import MaxPoolLayer

def save_to_json(results, filename='results/cnn/results.json'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        json.dump([results], file, indent=4)

def one_hot_encode(y, num_classes):
    one_hot = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    one_hot[np.arange(y.shape[0]), y] = 1.0
    return one_hot

def load_and_preprocess_data_for_cnn(train_limit, test_limit, validation_limit):
    (X_train, y_train), (X_test, y_test) = load_mnist(train_limit, test_limit)

    # Pre-Processing (normalization)
    X_train = X_train.astype(np.float32) / 255.0
    X_test  = X_test.astype(np.float32) / 255.0

    # shape: (N, 1, 28, 28)
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test  = X_test.reshape(-1, 1, 28, 28)

    num_classes = 10
    y_train_onehot = one_hot_encode(y_train, num_classes)
    y_test_onehot  = one_hot_encode(y_test, num_classes)

    X_train_final = X_train[:validation_limit]
    y_train_final = y_train_onehot[:validation_limit]
    X_val         = X_train[validation_limit:]
    y_val         = y_train_onehot[validation_limit:]

    return X_train_final, y_train_final, X_val, y_val, X_test, y_test_onehot

def build_and_train_cnn_model_with_config(config, loss_fun=SoftmaxCrossEntropy()):
    X_train, y_train_onehot, X_val, y_val, X_test, y_test_onehot = load_and_preprocess_data_for_cnn(
        config['train_dimension'],
        config['test_dimension'],
        config['validation_dimension'])
    nn = Convolutional(loss_fun)

    input_channels = 1
    for conv_conf in config.get("conv_layers", []):
        conv_layer = ConvLayer(
            input_channels=input_channels,
            num_filters=conv_conf["filters"],
            kernel_size=conv_conf["kernel_size"],
            stride=conv_conf["stride"],
            padding=conv_conf["padding"],
            activation=conv_conf["activation"]
        )
        nn.add_conv_layer(conv_layer)
        input_channels = conv_conf["filters"]

    for pool_conf in config.get("max_pool_layers", []):
        pool_layer = MaxPoolLayer(
            kernel_size=pool_conf["kernel_size"],
            stride=pool_conf["stride"]
        )
        nn.add_pool_layer(pool_layer)

    dummy_input = np.zeros((1, 1, 28, 28), dtype=np.float32)
    out         = dummy_input
    for conv in nn.conv_layers:
        out = conv.forward(out)
    for pool in nn.pool_layers:
        out = pool.forward(out)
    flattened_size = int(np.prod(out.shape[1:]))  # Don't consider batch_size

    fc_layers_config = config.get("fc_layers", [])
    if not fc_layers_config:
        raise ValueError("Convolutional config must contain at least one fc_layer.")

    nn.fc_layer = []
    input_size = flattened_size
    for fc_conf in fc_layers_config:
        fc_layer = DenseLayer(
            input_size,
            fc_conf["neurons"],
            activation=fc_conf["activation"],
            learning_rate=0.001
        )
        nn.fc_layers.append(fc_layer)
        input_size = fc_conf["neurons"]

    epochs        = config["epochs"]
    batch_size    = config["batch_size"]
    learning_rate = 0.001

    nn.train(X_train, y_train_onehot, epochs, X_val, y_val, learning_rate, batch_size)

    train_accuracy      = nn.arithmetic_mean_accuracy(X_train, y_train_onehot)
    test_accuracy       = nn.arithmetic_mean_accuracy(X_test, y_test_onehot)
    validation_accuracy = nn.arithmetic_mean_accuracy(X_val, y_val)

    result = {
        'conv_layers': config['conv_layers'],
        'max_pool_layers': config['max_pool_layers'],
        'fc_layers': config['fc_layers'],
        'train_dimension': config['train_dimension'],
        'test_dimension': config['test_dimension'],
        'validation_dimension': config['validation_dimension'],
        'epochs': epochs,
        'batch_size': batch_size,
        'train_accuracy': "{:.2f}".format(train_accuracy * 100),
        'test_accuracy': "{:.2f}".format(test_accuracy * 100),
        'validation_accuracy': "{:.2f}".format(validation_accuracy * 100)
    }
    return result

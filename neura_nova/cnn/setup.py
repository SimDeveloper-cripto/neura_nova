import numpy as np

from .network import Convolutional

from ..data import load_mnist
from ..ff.layer import DenseLayer
from ..loss import SoftmaxCrossEntropy

from .conv2D import Conv2D
from .pool_layer import MaxPoolLayer

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
    test_dimension = config['test_dimension']

    X_train, y_train_onehot, X_val, y_val, X_test, y_test_onehot = load_and_preprocess_data_for_cnn(
        config['train_dimension'],
        test_dimension,
        config['validation_dimension'])
    nn = Convolutional(loss_fun)

    lr      = config['learning_rate']
    beta1   = config['beta1']
    beta2   = config['beta2']
    epsilon = config['epsilon']

    input_channels = 1
    for conv_conf in config.get("conv_layers", []):
        conv_layer = Conv2D(
            input_channels=input_channels,
            filter_number=conv_conf["filters"],
            kernel_size=conv_conf["kernel_size"],
            stride=conv_conf["stride"],
            activation_funct=conv_conf["activation"],
            learning_rate=lr,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon
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
            learning_rate=lr,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon
        )
        nn.fc_layers.append(fc_layer)
        input_size = fc_conf["neurons"]

    epochs        = config["epochs"]
    batch_size    = config["batch_size"]

    nn.train(X_train, y_train_onehot, epochs, X_val, y_val, batch_size)

    # Sulla base dei pesi migliori
    test_accuracy = nn.getAccuracy(X_test, y_test_onehot, test_dimension)

    result = {
        'conv_layers': config['conv_layers'],
        'max_pool_layers': config['max_pool_layers'],
        'fc_layers': config['fc_layers'],
        'train_dimension': config['train_dimension'],
        'test_dimension': config['test_dimension'],
        'validation_dimension': config['validation_dimension'],
        'epochs': epochs,
        'batch_size': batch_size,
        'test_accuracy': "{:.2f}".format(test_accuracy * 100)
    }
    return result
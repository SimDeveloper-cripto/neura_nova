import numpy as np

from .network import Convolutional

from ..data import load_mnist
from ..ff.layer import DenseLayer
from ..loss import SoftmaxCrossEntropy

from .conv_layer import ConvLayer
from .pool_layer import MaxPoolLayer

def one_hot_encode(y, num_classes):
    one_hot = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    one_hot[np.arange(y.shape[0]), y] = 1.0
    return one_hot

def load_and_preprocess_data_for_cnn(train_limit, test_limit):
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

    train_ratio = 0.8
    split_index = int(X_train.shape[0] * train_ratio)

    X_train_final = X_train[:split_index]
    y_train_final = y_train_onehot[:split_index]
    X_val         = X_train[split_index:]
    y_val         = y_train_onehot[split_index:]

    return X_train_final, y_train_final, X_val, y_val, X_test, y_test_onehot

def build_cnn_model(loss_fun=SoftmaxCrossEntropy()):
    nn = Convolutional(loss_fun)
    nn.add_conv_layer(ConvLayer(1, 6, 3, 1, 1, 'relu'))
    nn.add_pool_layer(MaxPoolLayer())

    d_input = np.zeros((1, 1, 28, 28), dtype=np.float32)
    out = d_input

    for conv in nn.conv_layers:
        out = conv.forward(out)
    for pool in nn.pool_layers:
        out = pool.forward(out)

    flattened_size = int(np.prod(out.shape[1:]))  # Don't consider batch_size
    nn.fc_layer    = DenseLayer(flattened_size, 10, activation='identity', learning_rate=0.001)
    return nn

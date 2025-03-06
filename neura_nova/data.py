# neura_nova/data.py

import os
import struct
import numpy as np


def load_mnist(train_limit, test_limit, path='../data/MNIST/raw'):
    def load_images(filename):
        with open(filename, 'rb') as f:
            magic_num, total, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(total, rows, cols)

        return images

    def load_labels(filename):
        with open(filename, 'rb') as f:
            magic_num, num = struct.unpack(">II", f.read(8))
            labels         = np.frombuffer(f.read(), dtype=np.uint8)

        return labels

    X_train = load_images(os.path.join(path, 'train-images-idx3-ubyte'))
    y_train = load_labels(os.path.join(path, 'train-labels-idx1-ubyte'))
    X_test  = load_images(os.path.join(path, 't10k-images-idx3-ubyte'))
    y_test  = load_labels(os.path.join(path, 't10k-labels-idx1-ubyte'))
    X_train, y_train = X_train[:train_limit], y_train[:train_limit]
    X_test,  y_test  = X_test[:test_limit],   y_test[:test_limit]

    return (X_train, y_train), (X_test, y_test)

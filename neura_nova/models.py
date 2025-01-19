# neura_nova/models.py

"""
Here we encapsulate two classes of artificial neural networks: feed forward and convolutional.
"""

import torch
import torch.nn as nn


# TODO: DOCUMENT
# TODO: WHAT IS THE MEANING OF "hidden_size"?

class FeedForward(nn.Module):
    """
    Feed-Forward neural network for MNIST.
    """

    def __init__(self, input_size=28*28, hidden_size=128, num_classes=10):
        """
        :param hidden_size: hyper-parameter to represent data complexity
        128 is usually a standard value
        """
        super(FeedForward, self).__init__()

        # Created 2 Layers (fully-connected)
        # Layer 1: Y = WX + B (or XW^T + B) TODO: IS IT THE SAME?
        self.layer1  = nn.Linear(input_size, hidden_size)
        self.relu    = nn.ReLU()

        # Layer 2
        self.layer2  = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Flattening: from (N, 1, 28, 28) to (N, 28*28)
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


# TODO: WORRY ABOUT THIS LATER
class Convolutional(nn.Module):
    """
    Convolutional neural network for MNIST.
    """

    def __init__(self, num_classes=10):
        super(Convolutional, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Linear(64 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

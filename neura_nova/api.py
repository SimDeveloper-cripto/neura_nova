# neura_nova/api.py

from neura_nova.data import get_loaders
from neura_nova.utils import show_image

from neura_nova.trainer import Trainer
from neura_nova.models import FeedForward, Convolutional

# TODO: TIDY UP

# Public Interface
class NeuraNova:
    """
    All Networks are fully-connected!
    """

    # TODO: __init__()?

    def CreateFeedForwardNetwork(self):
        return FeedForward()

    def CreateConvolutionalNetwork(self):
        return Convolutional()

    def InitTrainer(self, model, train_loader, test_loader, lr=0.001):
        return Trainer(model, train_loader, test_loader, lr=0.001)

    def getLoaders(self, dataset_path, batch_size):
        return get_loaders(dataset_path, batch_size)

    # Utility
    def showImage(self, batch, index):
        show_image(batch, index)

# neura_nova/data.py

"""
This module is dedicated to loading and preparing the MNIST dataset.
Here we encapsulate the code to create DataLoaders with the objective of centralizing
all data-related logic.
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# TODO: DEFINE WHAT WE MEAN BY "BATCH" SIZE
# TODO: DEFINE "mean" and "std deviation"
def get_loaders(dataset_path, batch_size):
    """
    Returns:
         tuple: (train_loader, test_loader)
    """

    # Convert images into tensors
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize
    ])

    train_dataset = datasets.MNIST(root=dataset_path, train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root=dataset_path, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

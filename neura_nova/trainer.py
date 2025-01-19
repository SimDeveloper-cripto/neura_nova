# neura_nova/trainer.py

"""
This module is dedicated to train models.
"""

# TODO: SAVE TRAIN INFO IN FILES

import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:
    def __init__(self, model, train_loader, test_loader, lr=0.001, device=None):
        """
        Args:
            device: device used to train (CPU or GPU).
        """
        self.model        = model
        self.train_loader = train_loader
        self.test_loader  = test_loader

        # TODO: USE GPU PLEASE
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        # Update model parameters
        # TODO: DO WE USE ADAM?
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    # TODO: DEFINE WHAT KIND OF TRAINING STRATEGY (ONLINE, BATCH, MINI-BATCH, ECC.)
    # Mini-Batch
    def train(self):
        """
        Execute just on epoch of training

        60.000/64 ruffly 938 batches for each epoch
        """
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss   = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            # Print training state every 100 bat
            if batch_idx % 100 == 0:
                print(f'Train batch: {batch_idx}, loss: {loss.item():.4f}')

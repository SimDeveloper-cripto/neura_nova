# neura_nova/loss.py

import numpy as np

class LossFunction:
    def forward(self, logits, labels):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

# TODO: E' CORRETTO UTILIZZARLA? LA USANO TUTTI I LAYER GIUSTO? NON SO SE E' CORRETTO
class SoftmaxCrossEntropyLoss(LossFunction):
    def __init__(self):
        self.probs = None
        self.labels = None

    def forward(self, logits, labels):
        # Softmax
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits     = np.exp(shifted_logits)
        self.probs     = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        self.labels    = labels

        # Cross-Entropy
        loss = -np.sum(labels * np.log(self.probs + 1e-9)) / logits.shape[0]
        return loss

    def backward(self):
        batch_size = self.labels.shape[0]
        return (self.probs - self.labels) / batch_size

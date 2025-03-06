# neura_nova/loss.py

import numpy as np

class LossFunction:
    def forward(self, logits, labels):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

class SoftmaxCrossEntropy(LossFunction):
    def __init__(self):
        self.probs      = None  # (num_classes, batch_size)
        self.labels     = None  # (num_classes, batch_size)
        self.batch_size = None

    def forward(self, logits, labels):
        shifted_logits = logits - np.max(logits, axis=0, keepdims=True)
        exp_logits     = np.exp(shifted_logits)
        self.probs     = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
        self.labels    = labels

        self.batch_size = logits.shape[1]
        log_p      = np.log(self.probs + 1e-9)

        loss       = -np.sum(labels * log_p) / self.batch_size
        return loss

    def backward(self):
        return (self.probs - self.labels) / self.batch_size
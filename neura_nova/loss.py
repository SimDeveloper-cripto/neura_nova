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
        """
        logits shape: (num_classes, batch_size)
        labels shape: (num_classes, batch_size) [one-hot]

        1) shift = logits - max(logits, axis=0) max over each column
        2) exponents
        3) normalize on axis=0
        4) Cross-Entropy: -1/batch_size * sum(labels * log(probs)
        """
        # Softmax
        shifted_logits = logits - np.max(logits, axis=0, keepdims=True)
        exp_logits     = np.exp(shifted_logits)
        self.probs     = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
        self.labels    = labels

        # Cross-Entropy
        self.batch_size = logits.shape[1]
        log_p      = np.log(self.probs + 1e-9)

        # Returns an average cross-entropy based on the batch
        loss       = -np.sum(labels * log_p) / self.batch_size
        return loss

    def backward(self):
        """
        dL/d(logits) = (probs - labels) / batch_size
        :return shape: (num_classes, batch_size)
        """
        return (self.probs - self.labels) / self.batch_size
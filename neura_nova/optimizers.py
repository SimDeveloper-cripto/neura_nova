# neura_nova/optimizers.py

import numpy as np

# TODO: ELIMINARE QUESTO FILE

class Optimizer:
    def update(self, params, grads):
        raise NotImplementedError("Optimizer must implement the update method.")

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, params, grads):
        for param, grad in zip(params, grads):
            param -= self.lr * grad

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = {}

    def update(self, params, grads):
        for i, (param, grad) in enumerate(zip(params, grads)):
            if i not in self.m:
                self.m[i] = np.zeros_like(grad)
                self.v[i] = np.zeros_like(grad)
                self.t[i] = 0
            self.t[i] += 1
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t[i])
            v_hat = self.v[i] / (1 - self.beta2 ** self.t[i])
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

class RProp(Optimizer):
    def __init__(self, eta_plus=1.2, eta_minus=0.5, delta_max=50, delta_min=1e-6):
        self.eta_plus   = eta_plus
        self.eta_minus  = eta_minus
        self.delta_max  = delta_max
        self.delta_min  = delta_min
        self.prev_grads = {}
        self.deltas     = {}

    def update(self, params, grads):
        for i, (param, grad) in enumerate(zip(params, grads)):
            if i not in self.prev_grads:
                self.prev_grads[i] = np.zeros_like(grad)
                self.deltas[i] = np.ones_like(grad) * 0.1

            sign_change = self.prev_grads[i] * grad
            increase = sign_change > 0
            decrease = sign_change < 0
            same = sign_change == 0

            self.deltas[i][increase] = np.minimum(self.deltas[i][increase] * self.eta_plus, self.delta_max)
            self.deltas[i][decrease] = np.maximum(self.deltas[i][decrease] * self.eta_minus, self.delta_min)

            # Don't update parameters if the sign changed
            grad_update = np.where(decrease, 0, np.sign(grad))
            param -= self.deltas[i] * grad_update
            self.prev_grads[i] = grad

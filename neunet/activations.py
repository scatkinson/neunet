import numpy as np
from abc import ABC


class Activation(ABC):
    def __init__(self):
        pass

    def eval(self, x):
        pass

    def eval_derivative(self, **kwargs):
        pass


class ReLu(Activation):
    def __init__(self):
        super().__init__()

    def eval(self, x):
        return np.maximum(0, x)

    def eval_derivative(self, dA_curr, Z_curr):
        dZ = np.array(dA_curr, copy=True)
        dZ[Z_curr <= 0] = 0
        return dZ


class SoftMax(Activation):
    def __init__(self):
        super().__init__()

    def eval(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def eval_derivative(self, dA_curr, Z_curr):
        pass


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def eval(self, x):
        return 1 / (1 + np.exp(-x))

    def eval_derivative(self, dA_curr, Z_curr):
        sig = self.eval(Z_curr)
        return dA_curr * sig * (1 - sig)


class Tanh(Activation):
    def __init__(self):
        super().__init__()

    def eval(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


class HardTanh(Activation):
    def __init__(self):
        super().__init__()

    def eval(self, x):
        if x < -1:
            return -1
        elif x < 1:
            return x
        else:
            return 1

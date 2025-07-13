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


class SoftMax(Activation):
    def __init__(self):
        super().__init__()

    def eval(self, x):
        return np.exp(x) / np.sum(np.exp(x))


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def eval(self, x):
        return 1 / (1 + np.exp(-x))


class Tanh(Activation):
    def __init__(self):
        super().__init__()

    def eval(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


class HardTang(Activation):
    def __init__(self):
        super().__init__()

    def eval(self, x):
        if x < -1:
            return -1
        elif x < 1:
            return x
        else:
            return 1

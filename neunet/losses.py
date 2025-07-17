import numpy as np
from abc import ABC

import neunet.constants as const


class Loss(ABC):

    def __init__(self):
        pass

    def eval(self, y: np.array, y_hat: np.array):
        pass

    def eval_derivative(self, y: np.array, y_hat: np.array):
        pass


class CrossEntropy(Loss):
    def __init__(self):
        super().__init__()

    def eval(self, y: np.array, y_hat: np.array):
        return -np.mean(y * np.log(y_hat + const.CROSSENTROPY_PAD))

    def eval_derivative(self, y: np.array, y_hat: np.array):
        return -(y / (y_hat + const.CROSSENTROPY_PAD) * len(y))


class MSE(Loss):
    def __init__(self):
        super().__init__()

    def eval(self, y: np.array, y_hat: np.array):
        return np.mean((y - y_hat) ** 2)

    def eval_derivative(self, y: np.array, y_hat: np.array):
        return -2 * (y - y_hat) / len(y)


class RMSE(Loss):
    def __init__(self):
        super().__init__()

    def eval(self, y: np.array, y_hat: np.array):
        return np.sqrt(np.mean((y - y_hat) ** 2))

    def eval_derivative(self, y: np.array, y_hat: np.array):
        return -(y - y_hat) / self.eval(y, y_hat)


class LogLoss(Loss):
    def __init__(self):
        super().__init__()

    def eval(self, y: np.array, y_hat: np.array):
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def eval_derivative(self, y: np.array, y_hat: np.array):
        return (1 - y) / (1 - y_hat) - y / y_hat

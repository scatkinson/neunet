import numpy as np


def cross_entropy(y: np.array, y_hat: np.array):
    return -np.mean(y * np.log(y_hat))


def mse(y: np.array, y_hat: np.array):
    return np.mean((y - y_hat) ** 2)


def rmse(y: np.array, y_hat: np.array):
    return np.sqrt(mse(y, y_hat))

import numpy as np
from typing import Callable
from abc import ABC

from neunet.layer_config import LayerConfig
from neunet.activations import *


class Layer(ABC):
    def __init__(self):
        pass

    def forward(self, **kwargs):
        pass

    def backward(self, **kwargs):
        pass


class DenseLayer(Layer):

    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons
        self.activation = self.get_activation()

    def get_activation(self):
        activation = ReLu()
        return activation

    def forward(
        self,
        inputs: np.array,
        weights: np.array,
        bias: np.array,
    ):
        Z_curr = np.dot(weights, inputs) + bias
        A_curr = self.activation.eval(Z_curr)
        return A_curr, Z_curr

    def backward(
        self,
        dA_curr: np.array,
        W_curr: np.array,
        b_curr: np.array,
        Z_curr: np.array,
        A_prev: np.array,
    ):
        m = A_prev.shape[1]
        dZ_curr = self.activation.eval_derivative(dA_curr, Z_curr)
        dW_curr = np.dot(dZ_curr, A_prev.T) / m
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr

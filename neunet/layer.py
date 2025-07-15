import numpy as np
from typing import Callable
from abc import ABC

from neunet.activations import *
import neunet.constants as const

activation_dict = {
    const.RELU_STR: ReLu(),
    const.SIGMOID_STR: Sigmoid(),
    const.SOFTMAX_STR: SoftMax(),
    const.TANH_STR: Tanh(),
    const.HARDTANH_STR: HardTanh(),
}


class Layer(ABC):

    num_neurons: int
    activation: Activation

    def __init__(self):
        pass

    def forward(self, **kwargs):
        pass

    def backward(self, **kwargs):
        pass


class DenseLayer(Layer):

    def __init__(self, num_neurons: int, activation_name: str):
        super().__init__()
        self.num_neurons = num_neurons
        self.activation = activation_dict[activation_name]

    def forward(
        self,
        inputs: np.array,
        weights: np.array,
        bias: np.array,
    ):
        Z_curr = np.dot(inputs, weights.T) + bias
        A_curr = self.activation.eval(Z_curr)
        return A_curr, Z_curr

    def backward(
        self,
        dA_curr: np.array,
        W_curr: np.array,
        Z_curr: np.array,
        A_prev: np.array,
    ):
        dA_curr = dA_curr.reshape(Z_curr.shape)
        dZ_curr = self.activation.eval_derivative(dA_curr=dA_curr, Z_curr=Z_curr)
        dW_curr = np.dot(dZ_curr.T, A_prev)
        db_curr = np.sum(dZ_curr, axis=0, keepdims=True)
        dA_prev = np.dot(dZ_curr, W_curr)

        return dA_prev, dW_curr, db_curr

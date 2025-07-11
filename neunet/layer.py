import numpy as np
from typing import Callable

from neunet.layer_config import LayerConfig


class Layer:

    def __init__(self, num_neurons):
        self.num_neurons = num_neurons

    def forward(
        self, inputs: np.array, weights: np.array, bias: np.array, activation: Callable
    ):
        Z_curr = np.dot(weights, inputs) + bias
        A_curr = activation(Z_curr)
        return A_curr, Z_curr

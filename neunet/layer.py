import numpy as np
from typing import Callable, Union
from abc import ABC

from neunet.activations import *
import neunet.constants as const
from neunet.util import correlate2d_valid, correlate2d_full

activation_dict = {
    const.IDENTITY_STR: Identity(),
    const.RELU_STR: ReLu(),
    const.SIGMOID_STR: Sigmoid(),
    const.SOFTMAX_STR: SoftMax(),
    const.TANH_STR: Tanh(),
    const.HARDTANH_STR: HardTanh(),
}


class Layer(ABC):

    input_shape: Union[tuple, int]
    activation: Activation
    filter_shape: tuple
    output_shape: tuple

    def __init__(self):
        pass

    def forward(self, **kwargs):
        pass

    def backward(self, **kwargs):
        pass


class DenseLayer(Layer):

    def __init__(self, input_shape: int, activation_name: str):
        super().__init__()
        self.input_shape = input_shape
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


class ConvolutionalLayer(Layer):
    def __init__(self, input_shape, filter_size, num_filters, activation_name):
        super().__init__()
        self.input_shape = input_shape
        self.input_len, self.input_height, self.input_width = input_shape
        self.filter_size = filter_size
        self.num_filters = num_filters
        # currently only coded for ReLu activation
        if activation_name != const.RELU_STR:
            m = f"ConvoluationalLayer activation_name value must be {const.RELU_STR}."
            raise ValueError(m)
        self.activation = activation_dict[activation_name]

        self.filter_shape = (self.num_filters, self.filter_size, self.filter_size)
        self.output_shape = (
            self.input_len,
            self.num_filters,
            self.input_height - self.filter_size + 1,
            self.input_width - self.filter_size + 1,
        )

    def forward(self, inputs, weights, bias):
        output = np.zeros(self.output_shape)
        for idx in range(inputs.shape[0]):
            for i in range(self.num_filters):
                output[idx, i] = correlate2d_valid(inputs[idx], weights[i])
        output += bias
        output = self.activation.eval(output)
        return output

    def backward(
        self,
        dA_curr: np.array,
        Z_curr: np.array,
        W_curr: np.array,
        A_prev: np.array,
    ):
        dA_prev = np.zeros_like(A_prev)
        dW_curr = np.zeros_like(W_curr)

        for idx in range(A_prev.shape[0]):
            for i in range(self.num_filters):
                dW_curr[i] += correlate2d(A_prev[idx], dA_curr[i])
                dA_prev += correlate2d(dA_curr[idx, i], W_curr[i])

        db_curr = np.sum(dA_curr, axis=0, keepdims=True)

        return (
            dA_prev,
            dW_curr,
            db_curr,
        )


class MaxPool(Layer):
    def __init__(self, pool_size, num_channels):
        super().__init__()
        self.pool_size = pool_size
        self.num_channels = num_channels
        self.input_height = 0
        self.input_width = 0
        self.output_height = 0
        self.output_width = 0

    def forward(self, inputs: np.array):
        self.input_len, self.num_channels, self.input_height, self.input_width = (
            inputs.shape
        )
        self.output_height = self.input_height // self.pool_size
        self.output_width = self.input_width // self.pool_size

        # Determining the output shape
        output = np.zeros((self.num_channels, self.output_height, self.output_width))

        # Iterating over different channels
        for c in range(self.num_channels):
            # Looping through the height
            for i in range(self.output_height):
                # looping through the width
                for j in range(self.output_width):

                    # Starting postition
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size

                    # Ending Position
                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size

                    # Creating a patch from the input data
                    patch = input_data[c, start_i:end_i, start_j:end_j]

                    # Finding the maximum value from each patch/window
                    output[c, i, j] = np.max(patch)

        return output

    def backward(self, A_prev, dA_curr, **kwargs):
        dA_prev = np.zeros_like(A_prev)
        dW_curr = np.zeros_like(A_prev)
        db_curr = np.zeros_like(A_prev)

        for c in range(self.num_channels):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size

                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size
                    patch = A_prev[c, start_i:end_i, start_j:end_j]

                    mask = patch == np.max(patch)

                    dA_prev[c, start_i:end_i, start_j:end_j] = dA_curr[c, i, j] * mask

        return (dA_prev, dW_curr, db_curr)

import numpy as np
import pandas as pd
import logging

from neunet.layer import *
from neunet.losses import *
import neunet.constants as const

loss_dict = {
    const.CROSSENTROPY_STR: CrossEntropy(),
    const.MSE_STR: MSE(),
    const.RMSE_STR: RMSE(),
    const.LOGLOSS_STR: LogLoss(),
}


class Network:

    def __init__(self, lr: float, n_epochs: int, loss_name: str, input_dim: int):
        self.lr = lr
        self.n_epochs = n_epochs
        self.layers: dict[int, Layer] = {}
        self.layer_cnt: int = 0
        self.architecture: dict = {}
        self.params: dict = {}
        self.memory: dict = {}
        self.gradients: dict = {}
        self.loss_name = loss_name
        self.loss: Loss = loss_dict[loss_name]
        self.loss_list = []
        self.input_dim: int = input_dim

    def add(self, layer: Layer):
        self.layers[self.layer_cnt] = layer
        self.layer_cnt += 1

    def compile(self):
        for key, layer in self.layers.items():
            if key == 0:
                self.architecture[key] = {
                    const.INPUT_DIM_KEY: self.input_dim,
                    const.OUTPUT_DIM_KEY: layer.num_neurons,
                    const.ACTIVATION_KEY: layer.activation,
                }
            else:
                self.architecture[key] = {
                    const.INPUT_DIM_KEY: self.layers[key - 1].num_neurons,
                    const.OUTPUT_DIM_KEY: layer.num_neurons,
                    const.ACTIVATION_KEY: layer.activation,
                }

    def init_weights(self, low, high):
        self.compile()

        for key, dim_dict in self.architecture.items():
            self.params[key] = {
                const.WEIGHT_KEY: np.random.uniform(
                    low=low,
                    high=high,
                    size=(
                        dim_dict[const.OUTPUT_DIM_KEY],
                        dim_dict[const.INPUT_DIM_KEY],
                    ),
                ),
                const.BIAS_KEY: np.zeros((1, dim_dict[const.OUTPUT_DIM_KEY])),
            }

    def forward(self, X):
        A_curr = X

        for key in self.layers.keys():
            A_prev = A_curr
            A_curr, Z_curr = self.layers[key].forward(
                inputs=A_prev,
                weights=self.params[key][const.WEIGHT_KEY],
                bias=self.params[key][const.BIAS_KEY],
            )
            self.memory[key] = {const.INPUTS_KEY: A_prev, const.Z_KEY: Z_curr}

        return A_curr

    def backprop(self, actual: np.array, predicted: np.array):
        actual = actual.reshape(predicted.shape)

        dA_prev = self.loss.eval_derivative(actual, predicted)
        for idx, layer in reversed(list(self.layers.items())):
            dA_curr = dA_prev

            A_prev = self.memory[idx][const.INPUTS_KEY]
            Z_curr = self.memory[idx][const.Z_KEY]
            W_curr = self.params[idx][const.WEIGHT_KEY]

            dA_prev, dW_curr, db_curr = layer.backward(
                dA_curr=dA_curr,
                W_curr=W_curr,
                Z_curr=Z_curr,
                A_prev=A_prev,
            )

            self.gradients[idx] = {
                const.DW_KEY: dW_curr,
                const.DB_KEY: db_curr,
            }

    def update(self):
        for key, layer in self.layers.items():
            self.params[key][const.WEIGHT_KEY] -= (
                self.lr * self.gradients[key][const.DW_KEY]
            )
            self.params[key][const.BIAS_KEY] -= (
                self.lr * self.gradients[key][const.DB_KEY]
            )

    def train(self, X, y, wts_low, wts_high):
        self.init_weights(wts_low, wts_high)

        for i in range(self.n_epochs):
            y_hat = self.forward(X)
            y_hat = y_hat.reshape(y.shape)
            loss_value = self.loss.eval(y, y_hat)
            self.loss_list.append(loss_value)

            self.backprop(actual=y, predicted=y_hat)

            self.update()

            if (i + 1) % 50 == 0:
                logging.info(f"\nEPOCH: {i+1}\n LOSS ({self.loss_name}): {loss_value}")

    def batch_train(self, X, y, wts_low, wts_high, num_batches):
        self.init_weights(wts_low, wts_high)

        for i in range(self.n_epochs):

            indices = np.arange(len(X))
            np.random.shuffle(indices)
            batches = []
            batch_length = len(X) // num_batches
            for j in range(num_batches):
                if j < num_batches - 1:
                    batches.append(indices[j * batch_length : (j + 1) * batch_length])
                else:
                    batches.append(indices[j * batch_length :])
            for batch in batches:
                y_hat = self.forward(X[batch])
                y_hat = y_hat.reshape(y[batch].shape)
                loss_value = self.loss.eval(y[batch], y_hat)
                self.loss_list.append(loss_value)

                self.backprop(actual=y[batch], predicted=y_hat)

                self.update()

            if (i + 1) % 50 == 0:
                logging.info(f"\nEPOCH: {i+1}\n LOSS ({self.loss_name}): {loss_value}")

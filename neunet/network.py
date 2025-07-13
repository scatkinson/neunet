import numpy as np
import pandas as pd

from neunet.network_config import NetworkConfig
from neunet.layer import DenseLayer
from neunet.losses import *
import neunet.constants as const


class Network:

    def __init__(self, conf: NetworkConfig):
        self.conf = conf
        np.random.seed(self.conf.seed)
        self.exog = self.get_exog()
        self.endog = self.get_endog()
        self.predicted = None
        self.layers: dict[Layer] = {}
        self.layer_cnt: int = 0
        self.architecture: dict = {}
        self.params: dict = {}
        self.memory: dict = {}
        self.gradients: dict = {}
        self.loss: Loss = self.get_loss()

    def get_exog(self):
        return {}

    def get_endog(self):
        return {}

    def get_loss(self):
        loss = LogLoss()
        return loss

    def add(self, layer: Layer):
        self.layers[self.layer_cnt] = layer
        self.layer_cnt += 1

    def compile(self):
        for key, layer in self.layers.items():
            if key == 0:
                self.architecture[key] = {
                    const.INPUT_DIM_KEY: self.exog.shape[1],
                    const.OUTPUT_DIM_KEY: layer.num_neurons,
                    const.ACTIVATION_KEY: self.conf.activations[key],
                }
            else:
                self.architecture[key] = {
                    const.INPUT_DIM_KEY: self.layers[key - 1].num_neurons,
                    const.OUTPUT_DIM_KEY: layer.num_neurons,
                    const.ACTIVATION_KEY: self.conf.activations[key],
                }

    def init_weights(self):
        self.compile()

        for key, dim_dict in self.architecture.items():
            self.params[key] = {
                const.WEIGHT_KEY: np.random.uniform(
                    low=-1,
                    high=1,
                    size=(
                        dim_dict[const.OUTPUT_DIM_KEY],
                        dim_dict[const.INPUT_DIM_KEY],
                    ),
                ),
                const.BIAS_KEY: np.zeros((1, dim_dict[const.OUTPUT_DIM_KEY])),
            }

    def forward(self):
        A_curr = self.exog

        for key in self.layers.keys():
            A_prev = A_curr
            A_curr, Z_curr = self.layers[key].forward(
                inputs=A_prev,
                weights=self.params[key][const.WEIGHT_KEY],
                bias=self.params[key][const.BIAS_KEY],
                activation=self.architecture[key][const.ACTIVATION_KEY],
            )
            self.memory[key] = {const.INPUTS_KEY: A_prev, const.Z_KEY: Z_curr}

        self.predicted = A_curr

    def backprop(self):
        self.endog.reshape(self.predicted.shape)

        dA_prev = self.loss.eval_derivative(self.endog, self.predicted)

        for layer_idx_prev, layer in reversed(list(enumerate(self.layers.items()))):
            layer_idx_curr = layer_idx_prev + 1
            dA_curr = dA_prev

            A_prev = self.memory[layer_idx_prev][const.INPUTS_KEY]
            Z_curr = self.memory[layer_idx_curr][const.Z_KEY]
            W_curr = self.params[layer_idx_curr][const.WEIGHT_KEY]
            b_curr = self.params[layer_idx_curr][const.BIAS_KEY]

            dA_prev, dW_curr, db_curr = layer.backward(
                dA_curr=dA_curr,
                W_curr=W_curr,
                b_curr=b_curr,
                Z_curr=Z_curr,
                A_prev=A_prev,
            )

            self.gradients[layer_idx_prev] = {
                const.DW_KEY: dW_curr,
                const.DB_KEY: db_curr,
            }

    def update(self):
        for key, layer in self.layers.items():
            self.params[key][const.WEIGHT_KEY] -= (
                self.conf.lr * self.gradients[key][const.DW_KEY]
            )
            self.params[key][const.BIAS_KEY] -= (
                self.conf.lr * self.gradients[key][const.DB_KEY]
            )

    def train(self):
        pass

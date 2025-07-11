import numpy as np
import pandas as pd

from neunet.network_config import NetworkConfig
from neunet.layer import Layer
import neunet.constants as const


class Network:

    def __init__(self, conf: NetworkConfig):
        self.conf = conf
        np.random.seed(self.conf.seed)
        self.exog = self.get_exog()
        self.endog = self.get_endog()
        self.layers: list[Layer] = []
        self.architecture: list[dict] = []
        self.params: list[dict] = []
        self.memory: list[dict] = []
        self.gradients = []

    def get_exog(self):
        return {}

    def get_endog(self):
        return {}

    def add(self, layer: Layer):
        self.layers.append(layer)

    def compile(self):
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                self.architecture.append(
                    {
                        const.INPUT_DIM_KEY: self.exog.shape[1],
                        const.OUTPUT_DIM_KEY: self.layers[idx].num_neurons,
                        const.ACTIVATION_KEY: self.conf.activations[idx],
                    }
                )
            else:
                self.architecture.append(
                    {
                        const.INPUT_DIM_KEY: self.layers[idx - 1].num_neurons,
                        const.OUTPUT_DIM_KEY: self.layers[idx].num_neurons,
                        const.ACTIVATION_KEY: self.conf.activations[idx],
                    }
                )

    def init_weights(self):
        self.compile()

        for dim_dict in self.architecture:
            self.params.append(
                {
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
            )

    def forward(self):
        A_curr = self.exog

        for layer, param, dim_dict in zip(self.layers, self.params, self.architecture):
            A_prev = A_curr
            A_curr, Z_curr = layer.forward(
                inputs=A_prev,
                weights=param[const.WEIGHT_KEY],
                bias=param[const.BIAS_KEY],
                activation=dim_dict[const.ACTIVATION_KEY],
            )
            self.memory.append({const.INPUTS_KEY: A_prev, const.Z_KEY: Z_curr})

        return A_curr

    def backprop(self):

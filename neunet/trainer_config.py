import pandas as pd

from neunet.config import Config, ConfigError
from neunet.layer import *


class TrainerConfig(Config):

    logging_path: str
    pipeline_id: str

    data_path: str
    target_col: str

    seed: int

    test_size: float

    dim_list: list[int]
    activation_list: list[str]
    lr: float
    n_epochs: int
    loss_str: str
    wts_low: float
    wts_high: float
    num_batches: int

    def __init__(self, config):
        super().__init__(config)
        if len(self.dim_list) != len(self.activation_list):
            m = "\ndim_list and activation_list must have same length,"
            m += f"\nbut they have lengths {len(self.dim_list)},"
            m += f" {len(self.activation_list)} respectively."
            raise ConfigError(m)

        self.data = pd.DataFrame()
        try:
            self.data = pd.read_csv(self.data_path)
        except UnicodeDecodeError:
            self.data = pd.read_pickle(self.data_path)
        self.data.reset_index(drop=True, inplace=True)

        self.layers: list[Layer] = self.get_layers()

    def get_layers(self):
        out = []
        if hasattr(self, "cnn_config"):
            out.append(
                ConvolutionalLayer(
                    input_shape=self.data.iloc[0, 1].shape,
                    filter_size=self.cnn_config[const.FILTER_SIZE_KEY],
                    num_filters=self.cnn_config[const.NUM_FILTERS_KEY],
                    activation_name=const.RELU_STR,
                )
            )
            out.append(
                MaxPool(
                    input_shape=out[-1].output_shape,
                    pool_size=self.cnn_config[const.POOL_SIZE_KEY],
                    num_channels=self.cnn_config[const.NUM_FILTERS_KEY],
                )
            )
        for num_neurons, activation_name in zip(self.dim_list, self.activation_list):
            out.append(
                DenseLayer(output_shape=num_neurons, activation_name=activation_name)
            )
        return out

    @property
    def required_config(self) -> dict:
        return {
            "logging_path": str,
            "pipeline_id": str,
            "data_path": str,
            "target_col": str,
            "seed": int,
            "test_size": float,
            "dim_list": list,
            "activation_list": list,
            "lr": float,
            "n_epochs": int,
            "loss_str": str,
            "wts_low": float,
            "wts_high": float,
        }

import pandas as pd
import os

from neunet.config import Config, ConfigError
from neunet.layer import *


class TrainerConfig(Config):

    logging_path: str
    pipeline_id: str

    data_path: str
    target_col: str

    seed: int

    test_size: float

    cnn_config: dict
    dim_list: list[int]
    activation_list: list[str]
    lr: float
    n_epochs: int
    loss_str: str
    wts_low: float
    wts_high: float
    num_batches: int

    save_output: bool

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

        # Save directory is the passed value if given, or the default value if none
        self.save_directory = getattr(
            self,
            "save_directory",
            os.path.join(
                const.DATA_STR,
                self.pipeline_id,
            ),
        )
        # If save path doesn't already exist, create the directory (needed to make new pipeline-named directories)
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory, exist_ok=True)
        self.y_hat_save_filename = "y_hat_" + self.pipeline_id
        self.y_hat_save_path = os.path.join(
            self.save_directory, self.y_hat_save_filename
        )
        self.X_test_save_filename = "X_test_" + self.pipeline_id + ".pkl"
        self.X_test_save_path = os.path.join(
            self.save_directory, self.X_test_save_filename
        )

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

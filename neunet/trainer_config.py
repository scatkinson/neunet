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

    def __init__(self, config):
        super().__init__(config)
        if len(self.dim_list) != len(self.activation_list):
            m = "\ndim_list and activation_list must have same length,"
            m += f"\nbut they have lengths {len(self.dim_list)},"
            m += f" {len(self.activation_list)} respectively."
            raise ConfigError(m)
        self.layers: list[Layer] = self.get_layers()

    def get_layers(self):
        out = []
        for num_neurons, activation_name in zip(self.dim_list, self.activation_list):
            out.append(
                DenseLayer(num_neurons=num_neurons, activation_name=activation_name)
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
        }

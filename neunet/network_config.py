from neunet.config import Config


class NetworkConfig(Config):

    data_path: str

    seed: int

    activations: list
    lr: float

    def __init__(self, config):
        super().__init__(config)

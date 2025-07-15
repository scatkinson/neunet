import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from abc import ABC
import logging

from neunet.trainer_config import TrainerConfig
from neunet.network import Network


class Trainer(ABC):

    def __init__(self, config: TrainerConfig):
        self.conf = config
        np.random.seed(self.conf.seed)

        self.data = pd.read_csv(self.conf.data_path)
        self.data.reset_index(drop=True, inplace=True)
        n_samples = len(self.data)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        split_point = int(n_samples * (1 - self.conf.test_size))

        train_indices = indices[:split_point]

        train_mask = np.zeros(n_samples, dtype=bool)

        train_mask[train_indices] = True

        self.data_train = self.data[train_mask]
        self.data_test = self.data[~train_mask]

        independent_cols = [
            col for col in self.data.columns if col != self.conf.target_col
        ]
        self.X_train = self.data_train[independent_cols]
        self.X_test = self.data_test[independent_cols]
        self.y_train = self.process_target(self.data_train[self.conf.target_col])
        self.y_test = self.process_target(self.data_test[self.conf.target_col])

        self.model = Network(
            lr=self.conf.lr,
            n_epochs=self.conf.n_epochs,
            loss_name=self.conf.loss_str,
            input_dim=self.X_train.shape[1],
        )
        for layer in self.conf.layers:
            self.model.add(layer)

    def train_model(self):
        self.model.train(self.X_train, self.y_train)

    def predict(self):
        y_hat = self.model.forward(self.X_test)
        return y_hat

    def test_model(self):
        pass

    def run_trainer(self):
        self.train_model()
        self.test_model()

    def process_target(self, y: pd.Series):
        return y.to_numpy()


class BinaryClassifierTrainer(Trainer):
    def __init__(self, config: TrainerConfig):
        super().__init__(config)

    def test_model(self):
        y_hat = self.predict()

        auc = roc_auc_score(self.y_test, y_hat)

        logging.info(f"AUC SCORE: {auc}")


class MultiClassTrainer(Trainer):

    def __init__(self, config: TrainerConfig):
        super().__init__(config)

    def process_target(self, y):
        y_arr = y.to_numpy()
        y_2d = np.zeros((y_arr.shape[0], 2), int)
        for idx in range(len(y_arr)):
            if y_arr[idx]:
                y_2d[idx] = np.array([0, 1])
            else:
                y_2d[idx] = np.array([1, 0])
        return y_2d

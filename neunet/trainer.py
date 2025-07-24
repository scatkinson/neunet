import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    r2_score,
    mean_squared_error,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from abc import ABC
import logging
import pickle

from neunet.trainer_config import TrainerConfig
from neunet.network import Network
from neunet.util import single_col_df_to_arr, class_to_indicator, indicator_to_class
import neunet.constants as const


class Trainer(ABC):

    def __init__(self, config: TrainerConfig):
        self.conf = config
        np.random.seed(self.conf.seed)

        self.data = self.conf.data
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
        if self.conf.save_output:
            logging.info(f"Saving X_test df at {self.conf.X_test_save_path}.")
            self.X_test.to_pickle(self.conf.X_test_save_path)
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
        self.model.train(
            self.X_train, self.y_train, self.conf.wts_low, self.conf.wts_high
        )

    def predict(self):
        y_hat = self.model.forward(self.X_test)
        return y_hat

    def test_model(self):
        y_hat = self.predict()
        return y_hat

    def save_y_hat(self, y_hat):
        logging.info(f"Saving y_hat to {self.conf.y_hat_save_path}.npy.")
        np.save(self.conf.y_hat_save_path, y_hat)

    def run_trainer(self):
        self.train_model()
        y_hat = self.test_model()
        if self.conf.save_output:
            self.save_y_hat(y_hat)

    def process_target(self, y: pd.Series):
        return y.to_numpy()


class BinaryClassifierTrainer(Trainer):
    def __init__(self, config: TrainerConfig):
        super().__init__(config)

    def test_model(self):
        y_hat = self.predict()

        auc = roc_auc_score(self.y_test, y_hat)

        logging.info(f"AUC SCORE: {auc}")
        return y_hat


class CNNTrainer(Trainer):

    def __init__(self, config: TrainerConfig):
        super().__init__(config)
        self.X_train = single_col_df_to_arr(self.X_train) / 255
        self.X_test = single_col_df_to_arr(self.X_test) / 255
        self.y_train_indicator = class_to_indicator(
            self.y_train, self.conf.dim_list[-1]
        )
        self.y_test_indicator = class_to_indicator(self.y_test, self.conf.dim_list[-1])

    def train_model(self):
        self.model.batch_train(
            self.X_train,
            self.y_train_indicator,
            self.conf.wts_low,
            self.conf.wts_high,
            self.conf.num_batches,
            cnn_wt_scale=self.conf.cnn_config[const.WT_SCALE_KEY],
            cnn_bias_scale=self.conf.cnn_config[const.BIAS_SCALE_KEY],
        )

    def test_model(self):
        y_hat = self.predict()
        y_hat_indicator = np.zeros_like(y_hat)
        for idx in range(y_hat.shape[0]):
            y_hat_indicator[idx, np.argmax(y_hat[idx])] = 1

        y_hat_classes = indicator_to_class(y_hat_indicator)

        cm = confusion_matrix(self.y_test, y_hat_classes)
        cr = classification_report(self.y_test, y_hat_classes)

        logging.info(f"\nCONFUSION MATRIX:\n{cm}\nCLASSIFICATION REPORT:\n{cr}")

        return y_hat_classes


class RegressionTrainer(Trainer):
    def __init__(self, config: TrainerConfig):
        super().__init__(config)
        self.xscaler = StandardScaler()
        self.yscaler = StandardScaler()

    def train_model(self):

        self.X_train = self.xscaler.fit_transform(self.X_train)
        self.y_train = self.y_train.reshape(-1, 1)
        self.y_train = self.yscaler.fit_transform(self.y_train)
        self.y_train = self.y_train.reshape(-1)

        self.model.batch_train(
            self.X_train,
            self.y_train,
            self.conf.wts_low,
            self.conf.wts_high,
            self.conf.num_batches,
        )

    def test_model(self):
        self.X_test = self.xscaler.transform(self.X_test)
        self.y_test = self.y_test.reshape(-1, 1)
        self.y_test = self.yscaler.transform(self.y_test)
        self.y_test = self.y_test.reshape(-1)
        y_hat = self.predict()
        r2 = r2_score(self.y_test, y_hat)
        logging.info(f"R2 Score: {r2}.")
        return y_hat

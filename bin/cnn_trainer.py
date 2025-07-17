from neunet.trainer_config import TrainerConfig
from neunet.trainer import CNNTrainer
from neunet.configamend import configamend

from neunet import log_wu

import argparse
import logging
import yaml
import sys

parser = argparse.ArgumentParser(
    prog="cnn_trainer",
    description="Trains a regression neural net",
    epilog="",
)

parser.add_argument(
    "-c",
    "--config",
    required=True,
    help="Path of config file, e.g. from working directory: bin/conf/cnn_test_config.yml",
)

args = parser.parse_args()


def main(conf: TrainerConfig):
    ct = CNNTrainer(conf)
    ct.run_trainer()


if __name__ == "__main__":
    with open(args.config, mode="r") as f:
        config = configamend(yaml.safe_load(f))
    conf = TrainerConfig(config)
    attrs = vars(conf)
    logging.info("Namespace: " + ", ".join("%s: %s" % item for item in attrs.items()))

    logging.info(f"Running main() of {__file__}")
    logging.info(f"Log will appear at {conf.logfile}")

    log_wu.run_with_logging(main, conf, logfile=conf.logfile)

#!/bin/python3
"""
TauClassifier
_______________________________________________________________
Main steering script for running the Tau Classifier
Example usages:
python3 tauclassifier.py train
python3 tauclassifier.py test -weights=network_weights/weights-20.h5
python3 tauclassifier.py scan -lr_range 5e-4 1e-1 10
"""

import os
import sys
import logger
log = logger.get_logger(__name__)
import run
import hydra
import matplotlib
from omegaconf import DictConfig
import tensorflow as tf


RUN_DICT = {'train': run.train,
            'evaluate': run.evaluate,
            'visualise': run.visualise,
            }

def invalid_run_mode(cfg: DictConfig) -> None:
    log.fatal(f"Run mode: {cfg.run_mode} not recognised! Available modes are {RUN_DICT.keys()}")
    sys.exit(1)

@hydra.main(config_path="config", config_name="config")
def unified_tau_classifier(config : DictConfig) -> None:

    # Setup enviroment from config
    if not config.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Run script
    RUN_DICT.get(config.run_mode, invalid_run_mode)(config)
    sys.exit(0)

if __name__ == "__main__":
    unified_tau_classifier()

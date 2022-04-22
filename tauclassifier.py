#!/bin/python3
"""
TauClassifier
_______________________________________________________________
Main steering script for running the Tau Classifier
Example usages:
python3 tauclassifier.py batch_size=524
python3 tauclassifier.py run_mode=evaluate
"""

import os
import sys
import glob
import logger
log = logger.get_logger(__name__)
import run
import hydra
import matplotlib
from omegaconf import DictConfig
import tensorflow as tf

# Dictionary of run_mode arguements mapped to corresponding function
RUN_DICT = {'train': run.train,
            'evaluate': run.evaluate,
            'visualise': run.visualise,
            }

def invalid_run_mode(cfg: DictConfig) -> None:
    """
    Helper function to gracefully exit program if run_mode arguement is invalid
    args:
        cfg: DictConfig - Hydra config object
    returns:
        None
    """
    log.fatal(f"Run mode: {cfg.run_mode} not recognised! Available modes are {RUN_DICT.keys()}")
    sys.exit(1)

@hydra.main(config_path="config", config_name="config")
def unified_tau_classifier(config: DictConfig) -> None:
    """
    Main steering script
    args:
        config: DictConfig - A dictionary-like config object created and passed to main() automagically by Hydra
    returns:
        None
    """

    # Setup enviroment from config
    if not config.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Check that files can be found
    tau_files = glob.glob(config.TauFiles)
    jet_files = glob.glob(config.FakeFiles)
    if len(tau_files) == 0 or len(jet_files) == 0:
        log.fatal("No input data found! Please check filepaths in config/config.yaml")
        sys.exit(1)

    # Run selected script. If arg invaild exit 
    RUN_DICT.get(config.run_mode, invalid_run_mode)(config)
    sys.exit(0)

if __name__ == "__main__":
    unified_tau_classifier()

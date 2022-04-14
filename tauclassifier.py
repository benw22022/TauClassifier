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

import logger
log = logger.get_logger(__name__)
import os
import sys
import argparse
import tensorflow as tf
import matplotlib
import hydra
from omegaconf import DictConfig, OmegaConf
from run import train

@hydra.main(config_path="config", config_name="config")
def unified_tau_classifier(cfg : DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))

    # args = OmegaConf.to_yaml(cfg)

    log.error("Test")
    train(cfg)

if __name__ == "__main__":
    unified_tau_classifier()

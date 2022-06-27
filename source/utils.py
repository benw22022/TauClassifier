"""
Utility functions
"""

import logger
log = logger.get_logger(__name__)
import os
import glob
import random
import math
import uproot
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa
import focal_loss
from omegaconf import DictConfig
from typing import Tuple, List

def get_files(config: DictConfig, file_type: str) -> Tuple[List[str]]:
    """
    Parse config file to get list of test, train, val files
    args:
        config: DictConfig - Hydra config object
        file_type: str - Key to glob-able path that will give file list
    returns:
        train_files, val_files, test_files
    """
    
    # Get list of files
    files = glob.glob(config[file_type])
    log.debug(f"Found {len(files)} for {file_type} on path {config[file_type]}")
    
    # If requested, only take a fraction of the files
    if config.fraction < 1:
        log.info(f"Using {config.fraction * 100}% of data from {file_type}")
        random.seed(42)
        files = random.sample(files, math.ceil(len(files) * config.fraction))

    # Do test/train/val split
    train_files, test_files = train_test_split(files, test_size=config.TestSplit, random_state=config.RandomSeed)
    train_files, val_files = train_test_split(files, test_size=config.ValSplit, random_state=config.RandomSeed)
    
    log.debug(f"Training files for {file_type}:")
    [log.debug(file) for file in train_files]
    log.debug(f"Testing files for {file_type}:")
    [log.debug(file) for file in test_files]
    log.debug(f"Validation files for {file_type}:")
    [log.debug(file) for file in val_files]
    
    return train_files, val_files, test_files

def get_optimizer(config: DictConfig) -> tf.keras.optimizers.Optimizer:

    opt_dict = {'Adam': tf.keras.optimizers.Adam(config.learning_rate, epsilon=config.epsilon),
                'Nadam': tf.keras.optimizers.Nadam(config.learning_rate, epsilon=config.epsilon),
                'SGD': tf.keras.optimizers.SGD(learning_rate=0.01, momentum=config.momentum),
                'RMSProp': tf.keras.optimizers.RMSprop(config.learning_rate),
                'Ftrl': tf.keras.optimizers.Ftrl(config.learning_rate)
                }
    
    try:
        log.info(f"Optimizer: {config.optimizer}")
        return opt_dict[config.optimizer]
    except KeyError:
        log.error(f'Optimizer {config.optimizer} not recognised! Options are {list(opt_dict.keys())}')
        log.warn('Using default optimizer \'Adam\'')
        return opt_dict['Adam']


def get_loss(config: DictConfig, class_weight: List[float]) -> tf.keras.losses.Loss:
    """
    Parse 'loss' option in config to get loss function
    args:
        config: DictConfig - Hydra config object
        class_weight: 
    """
    
    loss_dict = {'categorical_crossentropy': tf.keras.losses.CategoricalCrossentropy(),
                 'sparse_categorical_crossentropy': tf.keras.losses.SparseCategoricalCrossentropy(),
                 'focal_loss': focal_loss.SparseCategoricalFocalLoss(gamma=config.gamma, class_weight=list(class_weight.values())),
                 'sigmoid_focal_crossentropy': tfa.losses.SigmoidFocalCrossEntropy()}
    
    try:
        log.info(f"Loss function: {config.loss}")
        return loss_dict[config.loss]
    except KeyError:
        log.error(f'Loss {config.loss} not recognised! Options are {list(loss_dict.keys())}')
        log.warn('Using default loss \'categorical_crossentropy\'')
        loss_dict['categorical_crossentropy']
        return loss_dict['Adam']

def get_number_of_events(files: List[str]) -> List[int]:
    """
    Gets the number of events in each class
    args: 
        files: List[str] - A list of filepaths to ntuples
    returns:
        class_breakdown: List[int] - A list containing the number of events belonging to each class
    """
    
    all_labels = uproot.concatenate(files, filter_name="TauClassifier_Labels", library='np')["TauClassifier_Labels"]
    all_labels = np.vstack(all_labels)
    class_breakdown = []
    for l in range(0, all_labels.shape[1]):
        class_breakdown.append(np.sum(all_labels[:, l]))
    return class_breakdown

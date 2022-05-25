"""
Utility functions
"""

import logger
log = logger.get_logger(__name__)
import glob
import random
import math
from sklearn.model_selection import train_test_split
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

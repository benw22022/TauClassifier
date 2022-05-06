"""
Utility functions
"""

import glob
import random
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
    
    # If requested, only take a fraction of the files
    if config.fraction < 100:
        files = random.sample(files, len(files) * config.fraction / 100)

    # Do test/train/val split
    train_files, test_files = train_test_split(files, test_size=config.TestSplit, random_state=config.RandomSeed)
    train_files, val_files = train_test_split(files, test_size=config.ValSplit, random_state=config.RandomSeed)
    return train_files, val_files, test_files

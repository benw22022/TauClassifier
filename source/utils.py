"""
Utility functions
"""

import glob
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig
from typing import Tuple, List

def get_files(config: DictConfig, file_type: str) -> Tuple[List[str]]:
    files = glob.glob(config[file_type])
    train_files, test_files = train_test_split(files, test_size=config.TestSplit, random_state=config.RandomSeed)
    train_files, val_files = train_test_split(files, test_size=config.ValSplit, random_state=config.RandomSeed)
    return train_files, val_files, test_files

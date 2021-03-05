import uproot
import tracemalloc  # Monitor memory usage
from sklearn import preprocessing
import time
from variables import input_variables, variables
from dataloader import DataLoader
from models import experimental_model
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
from DataGenerator import DataGenerator
import awkward as ak
import pandas as pd
import gc
from sys import getsizeof
from dataset_maker import *


if __name__ == "__main__":

    time_memory_logger = TimeMemoryMonitor()

    # Grab all root files in directory
    # TODO: Make config file for datasets
    signal_dir = "E:\\MxAODs\\TauID\\signal"
    background_dir = "E:\\MxAODs\\TauID\\background"
    tree_name = ":CollectionTree"

    signal_files = get_root_files(signal_dir, tree_name)
    background_files = get_root_files(background_dir, tree_name)
    all_files = signal_files + background_files

    # Make the datasets
    make_files(all_files, input_variables)
    print("Done: Made npz files")
    time_memory_logger.get_current_usage()

    # Shuffle the datasets
    multithread_shuffle(input_variables, 12)
    time_memory_logger.get_current_usage()
    print("Done: Data shuffling")



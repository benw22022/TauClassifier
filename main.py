import uproot
import tracemalloc  # Monitor memory usage
from sklearn import preprocessing
import time
from variables import input_variables, variables_dictionary
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


def get_number_of_folders(path):
    files = folders = 0
    for _, dirnames, _ in os.walk(path):
        # ^ this idiom means "we won't be using this value"
        folders += len(dirnames)
    return folders


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

    print(all_files)

    # Make the datasets
    #print("Starting dataset creation")
    #multithread_write(all_files, input_variables, batch_size=10000)
    #print("Done: Made npz files")
    time_memory_logger.get_current_usage()

    # Shuffle the datasets
    #multithread_shuffle(input_variables, 12)
    #time_memory_logger.get_current_usage()
    #print("Done: Data shuffling")

    n_batches = get_number_of_folders("data")

    X_train_idx = np.arange(n_batches)
    X_train_idx, X_test_idx = train_test_split(X_train_idx, test_size=0.25, random_state=123456)
    X_train_idx, X_val_idx = train_test_split(X_train_idx, test_size=0.25, random_state=123456)

    # Initialize Generators
    training_batch_generator = DataGenerator(X_train_idx, variables_dictionary)
    testing_batch_generator = DataGenerator(X_test_idx, variables_dictionary)
    validation_batch_generator = DataGenerator(X_val_idx, variables_dictionary)

    # Initialize Model
    jet_df = training_batch_generator.fill_dataframe("Jets", 0)
    cls_df = training_batch_generator.fill_dataframe("Clusters", 0)
    trk_df = training_batch_generator.fill_dataframe("Tracks", 0)
    shape_trk = jet_df.shape
    shape_cls = cls_df.shape
    shape_jet = trk_df.shape
    model = experimental_model(shape_trk, shape_cls, shape_jet, unroll=False)
    model.compile(optimizer="adam", loss="binary_crossentropy",  metrics=["accuracy"])
    model.summary()

    # Train Model
    history = model.fit(training_batch_generator, epochs=100, max_queue_size=4, use_multiprocessing=False)

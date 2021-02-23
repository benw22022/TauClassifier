"""
Main Code Body
"""

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


def get_files(directory, treename):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".root"):
                file_list.append(os.path.join(root, file) + treename)
    return file_list


if __name__ == "__main__":

    # Start counting time and memory usage
    start_time = time.time()
    tracemalloc.start()

    # Grab all root files in directory
    signal_dir = "E:\\MxAODs\\TauID\\signal"
    background_dir = "E:\\MxAODs\\TauID\\background"
    tree_name = ":CollectionTree"

    signal_files = get_files(signal_dir, tree_name)
    background_files = get_files(background_dir, tree_name)
    all_files = signal_files + background_files

    signal_data = DataLoader(signal_files, variables)
    background_data = DataLoader(background_files, variables)
    all_data = DataLoader(all_files, variables)

    print("Found %i signal files with %i events" % (len(signal_files), signal_data.length()))
    print("Found %i background files with %i events" % (len(background_files), background_data.length()))

    # Create train/test/val datasets from random indices and label data
    signal_labels = np.ones(signal_data.length())
    background_labels = np.zeros(background_data.length())
    all_labels = np.concatenate((signal_labels, background_labels))
    indices = np.array(list(range(0, all_data.length())))

    print(len(indices))
    print(len(all_labels))

    random.Random(1234).shuffle(indices)             # Random seeds *must* be the same
    random.Random(1234).shuffle(all_labels)

    X_train_idx, X_test_idx, y_train, y_test = train_test_split(indices, all_labels, test_size=0.25, random_state=42)
    X_train_idx, X_val_idx, y_train, y_val = train_test_split(X_train_idx, y_train, test_size=0.25, random_state=42)

    print("TRAIN: x len = {0}    y len = {1}".format(len(X_train_idx), len(y_train)))
    print("TEST: x len = {0}    y len = {1}".format(len(X_test_idx), len(y_test)))
    print("VAL: x len = {0}    y len = {1}".format(len(X_val_idx), len(y_val)))

    # Can now free RAM by deleting the dataloaders
    del signal_data
    del background_data
    del all_data

    # Initialize data generators
    training_batch_generator = DataGenerator(all_files, variables, X_train_idx, y_train, 100)
    testing_batch_generator = DataGenerator(all_files, variables, X_test_idx, y_test, 100)
    validation_batch_generator = DataGenerator(all_files, variables, X_val_idx, y_val, 100)

    shape_trk = (6, 7)
    shape_cls = (10, 8)
    shape_jet = (8,)

    # Train model
    model = experimental_model(shape_trk, shape_cls, shape_jet)
    model.compile(optimizer="adam", loss="binary_crossentropy",  metrics=["accuracy"])
    model.summary()

    model.fit(training_batch_generator,
              validation_data=validation_batch_generator,
              use_multiprocessing=True,
              workers=6)

    # Print memory usage and time taken
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    print("--- %s seconds ---" % (time.time() - start_time))
    tracemalloc.stop()

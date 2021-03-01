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
import awkward as ak
import pandas as pd

def get_files(directory, treename):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".root"):
                file_list.append(os.path.join(root, file) + treename)
    return file_list


def append_dataframes(filelist, variables_list):
    file1 = uproot.open(filelist[0]).arrays(variables_list, library="pd")
    print("Opened 1 file")
    if len(filelist) > 0:
        for i in range(1, len(filelist)):
            file2 = uproot.open(filelist[0]).arrays(variables_list, library="pd")
            file1 = file1.append(file1, ignore_index=True)
    return file1


def get_arrays(data, var_list):
    data_dict = {}
    for variable in var_list:
        data_dict = {**data_dict, **{variable : np.array(ak.to_list(data[variable]), dtype="O")}}
        print("Done: " + variable)
    return data_dict


def save_arrays(file_list, class_name, var_list):
    for variable in var_list:
        data = uproot.concatenate(file_list, filter_name=variable)
        np.save("data\\" + class_name + "_" + variable + '.npy', np.array(ak.to_list(data[variable]), dtype="O"))
        print("Done: " + variable)


def load_arrays(class_name, var_list):
    data_dict = {}
    for variable in var_list:
        data = np.load("data\\" + class_name + "_" + variable + '.npy')
        data_dict = {**data_dict, **{variable: data}}
        print("Loaded: " + variable)
    return data

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

    #signal_data = DataLoader(signal_files, variables)
    #background_data = DataLoader(background_files, variables)
    #all_data = DataLoader(all_files, variables)

    #print("Found %i signal files with %i events" % (len(signal_files), signal_data.length()))
    #print("Found %i background files with %i events" % (len(background_files), background_data.length()))

    #signal_larr = uproot.lazy(signal_files, filter_name=input_variables, step_size=10000, array_cache="16000 MB",
    #                        num_workers=12, begin_chunk_size=512*64)

    #background_larrr = uproot.lazy(signal_files, filter_name=input_variables, step_size=10000, array_cache="16000 MB",
    #                        num_workers=12, begin_chunk_size=512 * 64)

    print("saving signal data")
    save_arrays(signal_files, "signal", input_variables)
    print("saving background data")
    save_arrays(background_files, "background", input_variables)

    signal_dict = load_arrays("signal", input_variables)
    background_dict = load_arrays("background", input_variables)

    signal_labels = np.ones((len(signal_dict[input_variables[0]])))
    background_labels = np.zeros((len(background_dict[input_variables[0]])))
    y = np.concatenate((signal_labels, background_labels))

    def merge_dicts(dict1, dict2):
        out_dict = {}
        for key in dict1:
            out_dict[key] = np.concatenate((dict1[key], dict2[key]))
        return out_dict

    X_dict = merge_dicts(signal_dict, background_dict)
    print("Merging dicts")
    #X = pd.DataFrame.from_dict(X_dict, orient='index')
    #print("Making dataframes")
   # X = pd.Series(X_dict)
   # print(X)

   # X_train, X_test  = [i.to_dict() for i in train_test_split(X, train_size=0.7, random_state=42)]
   # X_train, X_val = [i.to_dict() for i in train_test_split(X_train, train_size=0.7, random_state=42)]

    y_train, y_test = train_test_split(y, test_size=0.25, random_state=42)
    y_train, y_val = train_test_split(y_train, test_size=0.25, random_state=42)
    print("Done y splitting")


#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
 #   X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    print("Doing x splitting")
    np.random.seed(42)

    for key in X_dict:
        print(X_dict[key])

        new_array = X_dict[key]
        np.random.shuffle(new_array)
        X_dict[key] = new_array
        print("Shuffled " + key)


    #length = len(X_dict['TauJetsAuxDyn.trk_pt'])
    #new_index = np.random.permutation(length)

    #train_val_index = new_index[:int(length * 0.8)]
    #test_index = list(set(new_index) - set(train_val_index))
    #X_train = {key: [value[i] for i in train_val_index] for key, value in X_dict.items()}
    #X_val = {key: [value[i] for i in test_index] for key, value in X_dict.items()}


    """
    signal_df["class"] = 1
    background_df["class"] = 0
    #all_data_df = signal_df.append(background_df, ignore_index=True)
    all_data_df = ak.concatenate((signal_df, background_df))
    print(all_data_df)
    #print(all_data_df.info())

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    print("--- %s seconds ---" % (time.time() - start_time))

    # Specify the data
    #X = all_data_df.drop("class")
    X = all_data_df[input_variables]
    print(type(all_data_df))


    # Specify the target labels and flatten the array
    #y = all_data_df[['class']]
    y = all_data_df['class']


    # Create train/test/val datasets from random indices and label data
    #signal_labels = np.ones(signal_data.length())
    #background_labels = np.zeros(background_data.length())
    #all_labels = np.concatenate((signal_labels, background_labels))
    #indices = np.array(list(range(0, all_data.length())))

    #print(len(indices))
    #print(len(all_labels))

    #random.Random(1234).shuffle(indices)             # Random seeds *must* be the same
    #random.Random(1234).shuffle(all_labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    print("TRAIN: x len = {0}    y len = {1}".format(len(X_train), len(y_train)))
    print("TEST: x len = {0}    y len = {1}".format(len(X_test), len(y_test)))
    print("VAL: x len = {0}    y len = {1}".format(len(X_val), len(y_val)))

    #X_train = X_train.drop("EventInfoAuxDyn.mcEventWeights", axis=1)
    #X_test = X_test.drop("EventInfoAuxDyn.mcEventWeights", axis=1)
    #X_val = X_val.drop("EventInfoAuxDyn.mcEventWeights", axis=1)

    # Can now free RAM by deleting the dataloaders
    #del signal_data
    #del background_data
    #del all_data

    # Initialize data generators
    #training_batch_generator = DataGenerator(all_files, variables, X_train_idx, y_train, 100)
    #testing_batch_generator = DataGenerator(all_files, variables, X_test_idx, y_test, 100)
    #validation_batch_generator = DataGenerator(all_files, variables, X_val_idx, y_val, 100)
    """


    shape_trk = (6, 7)
    shape_cls = (10, 8)
    shape_jet = (8,)

    # Train model
    model = experimental_model(shape_trk, shape_cls, shape_jet)
    model.compile(optimizer="adam", loss="binary_crossentropy",  metrics=["accuracy"])
    model.summary()

    #model.fit(training_batch_generator,
    #          validation_data=validation_batch_generator,
    #          use_multiprocessing=True,
    #          workers=6)

    print(type(X_train[0]))


    def apply_var_selection(awk_list, var_list):
        out_arr = []
        for i in range(0, len(awk_list)):
            out_arr.append(ak.to_numpy(awk_list[i][var_list]))
        out_arr = np.asarray(out_arr, dtype="float")
        return out_arr

    #X_train_trks = X_train[0][variables["Tracks"]]
    #X_train_cls = X_train[0][variables["Clusters"]]
    #X_train_jets = X_train[0][variables["Jets"]]

    X_train_trks = apply_var_selection(X_train, variables["Tracks"])
    X_train_cls = apply_var_selection(X_train, variables["Clusters"])
    X_train_jets = apply_var_selection(X_train, variables["Jets"])

    print(X_train_trks)

    #X_train_trks = np.asarray(X_train_trks).astype('object')
    #X_train_cls = np.asarray(X_train_cls).astype('object')
    #X_train_jets = np.asarray(X_train_jets).astype('object')

    history = model.fit([X_train_trks, X_train_cls, X_train_jets], y_train, epochs=10, batch_size=1000,
                        validation_data=(X_val, y_val), validation_freq=1,
                        verbose=1, shuffle=True)

    # Print memory usage and time taken
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    print("--- %s seconds ---" % (time.time() - start_time))
    tracemalloc.stop()

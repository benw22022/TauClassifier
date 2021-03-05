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


def get_files(directory, treename):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".root"):
                file_list.append(os.path.join(root, file) + treename)
    return file_list


def get_files_of_type_in_dir(dir, extn):
    file_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            #if file.endswith(extn):
            if extn in file:
                file_list.append(os.path.join(root, file))
    return file_list


def STLVector_to_list(STLVector_array):
    new_array = []
    for i in range(0, len(STLVector_array)):
        new_array.append(STLVector_array[i].tolist())
    return np.array(new_array, dtype=object)


def make_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass


def make_files(file_list, variable_list, sample_name, class_label, step_size=100000, prog_count=10):
    counter = 0
    for batch in uproot.iterate(file_list, filter_name=variable_list, step_size=step_size, library="np"):
        make_dir("data\\" + str(counter))
        if class_label == 1:
            class_labels = np.ones(len(batch[input_variables]))
        elif class_label == 0:
            class_labels = np.ones(len(batch[input_variables]))
        else:
            print("ERROR: class_label must be either 0 or 1")
            raise ValueError
        np.savez(f"data\\{sample_name}_{class_labels}_{counter}", class_labels)
        for variable in variable_list:
            array = STLVector_to_list(batch[variable])
            np.savez(f"data\\{sample_name}_{variable}_{counter}", array)
        if counter % prog_count == 0:
            print(f"Done batch {counter}")
        counter += 1

def pop_random(lst, rdm_seed=42):
    random.seed(rdm_seed)
    idx = random.randrange(0, len(lst))
    return lst.pop(idx)


def shuffle_data(variable, number_of_iterations=50, rdm_seed=42):

    for i in range(0, number_of_iterations):
        file_list = get_files_of_type_in_dir("data\\", variable+"_")

        pairs = []
        while file_list:
            rand1 = pop_random(file_list)
            rand2 = pop_random(file_list)
            pair = rand1, rand2
            pairs.append(pair)

        for pair in pairs:
            #print(pair)
            arr1 = np.load(pair[0], allow_pickle=True)["arr_0"]
            arr2 = np.load(pair[1], allow_pickle=True)["arr_0"]
            comb_arr = np.concatenate((arr1, arr2), axis=0)
            np.random.seed(rdm_seed)
            np.random.shuffle(comb_arr)
            split_arr = np.array_split(comb_arr, 2)
            new_arr1 = split_arr[0]
            new_arr2 = split_arr[1]
            np.savez(pair[0], new_arr1)
            np.savez(pair[1], new_arr2)
        print(f"Done {i+1} shuffles of {variable}")


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

    #make_files(signal_files, input_variables, "Gammatautau", 1)

    import multiprocessing
    import math
    nthreads = 12
    n_itr = 50
    agents = 12
    chunksize = math.ceil(len(input_variables) / 12)

    #with multiprocessing.Pool(processes=agents) as pool:
      #  result = pool.map(shuffle_data, input_variables, chunksize)


    arr = np.load("data\\180\\test_TauJetsAuxDyn.trk_nInnermostPixelHits_180.npz", allow_pickle=True)["arr_0"]
    print(arr)

    """
    write = False
    if write:
        for i in range(0, len(signal_files)):
            events = uproot.open(signal_files[i]).arrays(filter_name=input_variables, library="np")
            for variable in input_variables:
                classname = signal_files[i].split("\\")[-2]
                outfile = f"data\\{classname}_{variable}_{i}.npy"
                np.savez(outfile, events[variable])
                print("Saved file: " + outfile)
            del events

    data_dict = {}
    for variable in input_variables:
        file_list = get_files_of_type_in_dir("data\\", variable)
        arr_list = []
        for file in file_list:
            tmp_data = np.load(file, allow_pickle=True)
            arr_list.append(tmp_data["arr_0"])

        arr = np.concatenate(tuple(arr_list), axis=0)
        data_dict = {**data_dict, **{variable: arr}}
        print(f"Loaded: {variable}")

    print(data_dict)
"""
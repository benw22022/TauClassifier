"""
dataset maker
______________
Functions to make dataset from MxAODs
Code works by using uproot.iterate to load batches of data, extracting arrays containing variables of interest
TODO: Maybe make the shuffle stuff its own class - some of these methods probably don't ned to be directly exposed to
TODO: the user. Plus it might help set up the configs better like setting the number of shuffle iterations
"""

import numpy as np
import random
import os
import uproot
from variables import input_variables
import multiprocessing
import math
import tracemalloc  # Monitor memory usage
import time


def get_MxAODs(directory, treename):
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


def make_files(file_list, variable_list, sample_name="TauClassifier", step_size=100000, prog_count=10, overwrite=True):
    counter = 0
    for batch in uproot.iterate(file_list, filter_name=variable_list, step_size=step_size, library="np"):
        make_dir("data\\" + str(counter))
        for variable in variable_list:
            if overwrite is True:
                array = STLVector_to_list(batch[variable])
                np.savez(f"data\\{sample_name}_{variable}_{counter}", array)
            elif overwrite is False and os.path.isfile(f"data\\{sample_name}_{variable}_{counter}") is False:
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


def multithread_shuffle(variable_list, n_threads):
    n_itr = 50
    chunksize = math.ceil(len(input_variables) / n_threads)

    with multiprocessing.Pool(processes=n_threads) as pool:
        result = pool.map(shuffle_data, variable_list, chunksize)


# TODO: make muthithreaded shuffle for single variable operations

if __name__ == "__main__":

    # Start counting time and memory usage
    start_time = time.time()
    tracemalloc.start()

    # Grab all root files in directory
    # TODO: Make config file for datasets
    signal_dir = "E:\\MxAODs\\TauID\\signal"
    background_dir = "E:\\MxAODs\\TauID\\background"
    tree_name = ":CollectionTree"

    signal_files = get_MxAODs(signal_dir, tree_name)
    background_files = get_MxAODs(background_dir, tree_name)
    all_files = signal_files + background_files

    # Make the datasets
    make_files(all_files, input_variables)

    # Shuffle the datasets
    multithread_shuffle(input_variables, 12)

    # Print memory usage and time taken
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    print("--- %s seconds ---" % (time.time() - start_time))
    tracemalloc.stop()

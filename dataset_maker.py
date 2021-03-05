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
import multiprocessing
import math
from variables import input_variables
from utils import TimeMemoryMonitor


def get_root_files(directory, tree=""):
    """
    Function to grab all .root files in a specific directory and add them to a list. Additionally able to append a tree
    name to the file path for uproot to use
    :param directory: string - path to directory where root files are stored
    :param tree: string (optional) - Tree for uproot to load
    :return file_list: list - list of file paths to root files
    """
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".root"):
                file_list.append(os.path.join(root, file) + tree)
    return file_list


def get_files_of_type_in_dir(directory, extn):
    """
    Function to grab all files containing extn string
    :param directory: string - filepath to directory to search
    :param extn: string - file extension to search for
    :return file_list: list - list of filepaths to files with extn
    """
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if extn in file:
                file_list.append(os.path.join(root, file))
    return file_list


def STLVector_to_list(STLVector_array):
    """
    Function to convert the uproot::STLVector containers to nested lists
    See: https://uproot.readthedocs.io/en/latest/uproot.containers.STLVector.html
    :param STLVector_array: np.array - A numpy array whose elements contain uproot::STLVector containers
    :return: np.array - A numpy array with nested lists rather than uproot::STLVector containers
    """
    new_array = []
    for i in range(0, len(STLVector_array)):
        new_array.append(STLVector_array[i].tolist())
    return np.array(new_array, dtype=object)


def make_dir(path):
    """
    A small function to make directories and suppress errors when directory specified already exists
    :param path: string - path to directory to be made
    :return: None
    """
    try:
        os.mkdir(path)
    except OSError:
        pass


def make_files(file_list, variable_list, sample_name="TauClassifier", step_size=10000, prog_count=1, overwrite=True):
    """
    Function to read data from MxAODs into npz files. Reads in batches of events of size step_size, makes arrays,
    converts the uproot::STLVector containers and writes them to npz. Batches are stored in numerically ordered
    subdirectories within data
    :param file_list: list - list of file paths to root files with :Tree_name suffix
    :param variable_list: list - list of variables to read from root files
    :param sample_name: string (optional: default= "TauClassifier") - a name to give each array file
    :param step_size: int (optional: default = 10000) - number of events to read in for each batch
    :param prog_count: int (optional: default = 1) - sets the number of times to print message
    :param overwrite: bool (optional: default = True) - If True will overwrite existing files
    :return: None
    """
    counter = 0
    for batch in uproot.iterate(file_list, filter_name=variable_list, step_size=step_size, library="np"):
        make_dir("data\\" + str(counter))
        for variable in variable_list:
            if overwrite is True:
                array = STLVector_to_list(batch[variable])
                np.savez(f"data\\{counter}\\{sample_name}_{variable}_{counter}", array)
            elif overwrite is False and os.path.isfile(f"data\\{sample_name}_{variable}_{counter}") is False:
                array = STLVector_to_list(batch[variable])
                np.savez(f"data\\{counter}\\{sample_name}_{variable}_{counter}", array)
        if counter % prog_count == 0:
            print(f"Done batch {counter}")
        counter += 1


def pop_random(lst, rdm_seed=12):
    """
    Function to make help make random pairs
    :param lst: list - list of objects to pair up
    :param rdm_seed: int - random seed
    :return: a random object from list
    """
    random.seed(rdm_seed)
    idx = random.randrange(0, len(lst))
    return lst.pop(idx)


def shuffle_data(variable, number_of_iterations=50, rdm_seed=42):
    """
    A function to shuffle arrays belonging to a particular variable. The function works by making random pairs of files,
    then, for each pair: loading the arrays, concatenating them, shuffling, splitting back into two arrays and saving
    (overwriting contents of original arrays). This process is then repeated number_of_iterations times to ensure data
    is thoroughly randomised
    :param variable: string - variable whose arrays are to be shuffled
    :param number_of_iterations: int - number of times to repeat the array pairing and shuffling process. The larger the
                                       the number the better the shuffle
    :param rdm_seed: int - random seed (important! must be the same for each variable you shuffle!)
    :return: None
    """
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
    """
    Function to speed up shuffling of variable arrays by splitting the job into tasks to be done on multiple threads
    (Note: Not an expert in multi-threading in python I just happen to know that this method works)
    :param variable_list: list - list of variables to shuffle
    :param n_threads: int - number of threads to use
    :return:
    """
    # If we have more variables than threads we can group the jobs into chunks
    chunksize = math.ceil(len(input_variables) / n_threads)

    with multiprocessing.Pool(processes=n_threads) as pool:
        result = pool.map(shuffle_data, variable_list, chunksize)


# TODO: make muthithreaded shuffle for single variable operations

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

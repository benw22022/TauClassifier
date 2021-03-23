import uproot
from variables import variables_dictionary, input_variables
import numpy as np

"""
def generate_batches(files):
    for batch in uproot.iterate(files, step_size=10000, filter_name=input_variables, library='pd'):

        # Labelling
        truth_match_array = batch["TauJets.IsTruthMatched"].to_numpy()
        labels = np.zeros((len(truth_match_array)))
        for i in range(0, len(truth_match_array)):
            if truth_match_array[i] != 0:
                labels[i] = 1

        yield batch, labels
"""

"""
Keras Data Generator
"""

import keras
import uproot
import random
import pandas as pd
import awkward as ak
from multiprocessing import Pool
import math
import random
import threading
import time
import copy
import tensorflow as tf

class DataGenerator(keras.utils.Sequence):

    def __init__(self, file_dict, variables_dict, batch_size):
        print("Initializing DataGenerator")
        self._batch_size = batch_size
        self._variables_dict = variables_dict
        self._file_dict = file_dict

        self._num_events_dict = {}
        for file_type in file_dict:
            self._num_events_dict = {**self._num_events_dict, **{file_type: 0}}
        self._tot_num_events = 0

        # Get number of events in each file type (GammaTauTau, JZ1, JZ2 etc...)
        for file_type in file_dict:
            for batch in uproot.iterate(file_dict[file_type], step_size=10000, filter_name="TauJets.mcEventWeight"):
                self._num_events_dict[file_type] += len(batch["TauJets.mcEventWeight"])
                self._tot_num_events += len(batch["TauJets.mcEventWeight"])
        print("DataGenerator Initialized")


    def load_batch(self, file_type, idx):

        # Dataframe to save results to
        data_df = pd.DataFrame()

        # Load in a different number of events depending on the file type
        specific_batch_size = math.ceil(self._batch_size * self._num_events_dict[file_type] / self._tot_num_events)

        print(f"specific_batch_size = {specific_batch_size}")

        variables_list = []
        for variable_type in self._variables_dict:
            variable_type_list = self._variables_dict[variable_type]
            variables_list += variable_type_list
            print(variables_list)
        print("Made variables list")

        # Load array
        batches = [batch for batch in uproot.iterate(self._file_dict[file_type], step_size=specific_batch_size, filter_name=variables_list, library='np')]

        print("Got batches")
        print(batches)

        # Loop through array and convert numpy array to tensorflow ragged tensor
        for variable in variables_list:
            print(variable)
            print(type(variable))
            batch = batches[idx][variable]
            data_df[variable] = tf.ragged.constant(np.transpose(batch))

        # Assign labels
        if file_type == "Gammatautau":
            data_df["label"] = np.ones((specific_batch_size))
        else:
            data_df["label"] = np.zeros((specific_batch_size))

        return data_df

    def fill_dataframe(self, idx, var_list=None):
        combined_df = pd.DataFrame()
        for file_type in self._file_dict:
            print(f"Making dataframe for file type {file_type}")
            combined_df = combined_df.append(self.load_batch(file_type, idx))

        if var_list is not None:
            return combined_df[var_list]
        else:
            return combined_df

    def __len__(self):
        return len(self._batch_size)

    def __getitem__(self, idx):

        combined_df = self.fill_dataframe(idx)
        trk_df = combined_df[self._variables_dict["Tracks"]]
        cls_df = combined_df[self._variables_dict["Clusters"]]
        jet_df = combined_df[self._variables_dict["Jets"]]

        labels_df = combined_df["label"]
        combined_df.drop(["TauJets.IsTruthMatched"])

        return [tf.convert_to_tensor(trk_df), tf.convert_to_tensor(cls_df), tf.convert_to_tensor(jet_df)], tf.convert_to_tensor(labels_df)

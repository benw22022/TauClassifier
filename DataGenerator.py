"""
Keras Data Generator
"""

import keras
import numpy as np
import uproot
import random
import pandas as pd
import awkward as ak
from multiprocessing import Pool
import math
import random
import threading
import time
import tensorflow as tf

class DataGenerator(keras.utils.Sequence):

    def __init__(self, batch_indices, variables_dict):
        self._batch_indices = batch_indices
        self._variables_dict = variables_dict

    def _get_feature_arrays(self, feature):
        tmp_arr = []
        for variable in self._variables_dict[feature]:
            tmp_arr.append(variable)
        return tmp_arr

    def my_func(self, arg):
        arg = tf.convert_to_tensor(arg, dtype=tf.float32)
        return arg

    def fill_dataframe(self, feature, idx, sample_name="TauClassifier"):
        feature_df = pd.DataFrame()
        for feature_var in self._get_feature_arrays(feature):
            array_filepath = f"data\\{idx}\\{sample_name}_{feature_var}_{idx}.npz"
            with np.load(array_filepath, allow_pickle=True) as array_file:
                array = array_file["arr_0"]
                feature_df[feature_var] = tf.ragged.constant(array)
             #   print(f"{feature_var} is type {type(array)}   -> {array}")
                print(f"Loaded: {feature_var}")
            #print(feature_df.shape.rank)
        return feature_df

    def get_labels(self, idx, sample_name="TauClassifier", signal_channel_number=425200):

        array_filepath = f"data\\{idx}\\{sample_name}_EventInfoAuxDyn.mcChannelNumber_{idx}.npz"
        with np.load(array_filepath, allow_pickle=True) as array_file:
            channel_num_arr = array_file["arr_0"]
            labels_arr = np.ones(len(channel_num_arr))
            for i in range(0, len(channel_num_arr)):
                if channel_num_arr[i] != signal_channel_number:
                    labels_arr[i] = 0
        return labels_arr

    def __len__(self):
        return len(self._batch_indices)

    def __getitem__(self, idx):
        jet_df = self.fill_dataframe("Jets", idx)
        cls_df = self.fill_dataframe("Clusters", idx)
        trk_df = self.fill_dataframe("Tracks", idx)
        labels_arr = self.get_labels(idx)

        print("__getitem__: Made dataframes")

        # TODO: add weights

        return [tf.convert_to_tensor(jet_df), tf.convert_to_tensor(cls_df), tf.convert_to_tensor(trk_df)], labels_arr

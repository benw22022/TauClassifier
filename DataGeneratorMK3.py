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
from ClassLoader import ClassLoader


class DataGenerator(keras.utils.Sequence):

    def __init__(self, file_dict, variables_dict, batch_size):
        print("Initializing DataGenerator")
        self._batch_size = batch_size
        self._file_dict = file_dict
        self.data_classes = []

        self._variables_dict = variables_dict
        self._variables_list = []
        for _, variable_list in variables_dict.items():
            self._variables_list += variable_list

        for data_type, file_list in file_dict.items:
            self.data_classes.append(ClassLoader(data_type, file_list))

        self.total_num_events = 0
        for data_class in self.data_classes:
            self.total_num_events += data_class.num_events

        for data_class in self.data_classes:
            data_class.load_batches(self._variables_list, self.total_num_events, self._batch_size)

    def __len__(self):
        return len(self._batch_size)

    def load_batch(self, data_class, idx):
        batch = data_class.batches[idx]

        track_vars = [v.replace("TauTracks.", "") for v in self._variables_dict["Tracks"]]
        track_ak_arrays = ak.concatenate([batch["TauTracks"][var][:, :, None] for var in track_vars], axis=0)
        track_ak_padded_arrays = ak.pad_none(track_ak_arrays, 20, clip=True)
        track_np_arrays = ak.to_numpy(track_ak_padded_arrays)
        track_np_arrays = track_np_arrays.reshape(int(track_np_arrays.shape[0] / len(track_vars)), len(track_vars),
                                                  track_np_arrays.shape[1])

        cluster_vars = [v.replace("TauClusters.", "") for v in self._variables_dict["Clusters"]]
        cluster_ak_arrays = ak.concatenate([batch["TauClusters"][var][:, :, None] for var in cluster_vars], axis=0)
        cluster_ak_padded_arrays = ak.pad_none(cluster_ak_arrays, 15, clip=True)
        cluster_np_arrays = ak.to_numpy(cluster_ak_padded_arrays)
        cluster_np_arrays = cluster_np_arrays.reshape(int(cluster_np_arrays.shape[0] / len(cluster_vars)), len(cluster_vars),
                                                      cluster_np_arrays.shape[1])

        jet_vars = self._variables_dict["Jets"]
        jet_ak_arrays = ak.concatenate([batch[var][:] for var in jet_vars], axis=0)
        jet_np_arrays = ak.to_numpy(jet_ak_arrays)
        jet_np_arrays = jet_np_arrays.reshape(int(jet_np_arrays.shape[0] / len(jet_vars)), len(jet_vars))

        labels_np_array = data_class.batch_labels(idx)

        weight_np_array = ak.to_numpy(batch["TauJets.mcEventWeight"])

        return track_np_arrays, cluster_np_arrays, jet_np_arrays, labels_np_array, weight_np_array

    def __getitem__(self, idx):

        track_array = np.array([])
        cluster_array = np.array([])
        jet_array = np.array([])
        label_array = np.array([])
        weight_array = np.array([])

        for i in range(0, len(self.data_classes)):
            if i == 0:
                track_array, cluster_array, jet_array, label_array, weight_array = self.load_batch(
                    self.data_classes[i], idx)
            else:
                tmp_track_array, tmp_cluster_array, tmp_jet_array, tmp_label_array, tmp_weight_array = self.load_batch(
                    self.data_classes[i], idx)
                track_array = np.vstack((tmp_track_array, track_array))
                cluster_array = np.vstack((tmp_cluster_array, cluster_array))
                jet_array = np.vstack((tmp_jet_array, jet_array))
                label_array = np.vstack((tmp_label_array, label_array))
                weight_array = np.vstack((tmp_weight_array, weight_array))

        return [track_array, cluster_array, jet_array], label_array, weight_array

        #return [tf.convert_to_tensor(trk_df), tf.convert_to_tensor(cls_df), tf.convert_to_tensor(jet_df)], tf.convert_to_tensor(labels_df)

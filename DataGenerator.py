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

class DataGenerator(keras.utils.Sequence):

    def __init__(self, filename, features, indexes, labels, batch_size, step_size=100000, array_cache="8000 MB",):
        self._filename = filename
        self._features = features
        self._batch_size = batch_size
        self._labels = labels
        self._indexes = indexes

        # We read in a dict to
        tmp_arr = []
        for key in features:
            for i in range(0, len(features[key])):
                tmp_arr.append(features[key][i])
        self._lazy_array = uproot.lazy(filename, step_size=step_size, filter_name=tmp_arr, array_cache=array_cache,
                                       num_workers=12, begin_chunk_size=512*64)
        self._length = len(self._lazy_array)

    def _get_feature_arrays(self, batch, feature):
        tmp_arr = []
        for variable in self._features[feature]:
            tmp_arr.append[batch[variable]]
        return tmp_arr

    def __len__(self):
        return (np.ceil(self._length) / float(self._batch_size)).astype(np.int)

    def _make_batch(self, indices, obj):
        '''
        batch = []
        for variable in self._features[obj]:
            print("Make Batch: Processing variable -> " + variable)
            start_time = time.time()
            tmp_arr = []
            for i in range(0, len(indices)):
                if i % 10 == 0:
                    print("Done: %i events out of %i" % (i, self._batch_size))
                    print("--- %s seconds ---" % (time.time() - start_time))
                tmp_arr.append((self._lazy_array[variable, indices[i]]))
            batch.append(np.asarray(tmp_arr, dtype=object))

        return np.asarray(batch, dtype=object)
    '''
        batch = []
        for i in range(0, len(indices)):
            tmp_arr = []
            print("Make Batch: Processing index -> {0}".format(i))
            start_time = time.time()
            for variable in self._features[obj]:
                if i % 10 == 0:
                    print("Done: %i events out of %i" % (i, self._batch_size))
                    print("--- %s seconds ---" % (time.time() - start_time))
                tmp_arr.append((self._lazy_array[variable, indices[i]]))
        batch.append(np.asarray(tmp_arr, dtype=object))

        return np.asarray(batch, dtype=object)

    '''
    def _make_var_list(self, obj, indices, threads=12):

        def list_append(index_list, id, out_list):
            print("Process started on thread {0}".format(id))
            counter = 0
            for index in index_list:
                out_list.append(self._lazy_array[obj, index])
                counter +=1
                if counter % 10 == 0:
                    print("Done {0} Events".format(counter))

        size = self._batch_size
        jobs = []
        for i in range(0, threads):
            out_list = list()
            thread = threading.Thread(target=list_append(indices, i, out_list))
            jobs.append(thread)

        # Start the threads (i.e. calculate the random number lists)
        for j in jobs:
            j.start()

        # Ensure all of the threads have finished
        for j in jobs:
            j.join()
        return out_list

    def _make_batch(self, indices, variables):

        batch = []
        for variable in self._features[variables]:
            print("Making array for: " + variable)
            tmp_array = self._make_var_list(variable, indices)
            print("length of {0} {1} array".format(variable, len(tmp_array)))
            batch.append(tmp_array)
        return np.asarray(batch, dtype=object)
'''


    def __getitem__(self, idx):
        batch_indices = self._indexes[idx * self._batch_size: (idx + 1) * self._batch_size]
        #batch_x = self._make_batch(batch_indices) #self._lazy_array[[batch_indices]]
        #batch_weights = np.take_along_axis(self._lazy_array["EventInfoAuxDyn.mcEventWeights"], batch_indices, axis=0)  #self._lazy_array[[batch_indices]]["EventInfoAuxDyn.mcEventWeights"]
        '''
        @TODO
        Preprocessing goes here
        '''

        #batch_trk_x = pd.DataFrame(self._get_feature_arrays(batch_x, "Tracks"), columns=self._features["Tracks"])
        #batch_cls_x = pd.DataFrame(self._get_feature_arrays(batch_x, "Clusters"), columns=self._features["Clusters"])
        #batch_jet_x = pd.DataFrame(self._get_feature_arrays(batch_x, "Jets"), columns=self._features["Jets"])

        '''
        # Doesnt work 
        batch_trk_x = pd.DataFrame(self._make_batch(batch_indices, "Tracks"), columns=self._features["Tracks"]) # maybe dont make dataframes?
        batch_cls_x = pd.DataFrame(self._make_batch(batch_indices, "Clusters"), columns=self._features["Clusters"])
        batch_jet_x = pd.DataFrame(self._make_batch(batch_indices, "Jets"), columns=self._features["Jets"])
        batch_weights = pd.DataFrame(self._make_batch(batch_indices, "Weights"), columns=self._features["Weights"])
        batch_y = self._labels[idx * self._batch_size: (idx + 1) * self._batch_size] #np.take_along_axis(self._labels, batch_indices, axis=0)   #self.labels[[batch_indices]]
        '''
        batch_trk_x = self._make_batch(batch_indices, "Tracks")
        batch_cls_x = self._make_batch(batch_indices, "Clusters")
        batch_jet_x = self._make_batch(batch_indices, "Jets")
        batch_weights = self._make_batch(batch_indices, "Weights")
        batch_y = self._labels[idx * self._batch_size: (idx + 1) * self._batch_size] #np.take_along_axis(self._labels, batch_indices, axis=0)   #self.labels[[batch_indices]]

        return [batch_trk_x, batch_cls_x, batch_jet_x], batch_y, batch_weights

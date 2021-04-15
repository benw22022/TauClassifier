"""
DataLoader Class definition
________________________________________________________________________________________________________________________
A helper class to lazily load batches of training data for each type of data
"""

import uproot
import math
import numpy as np
from utils import logger
import pickle
import sys

class DataLoader:

    def __init__(self, data_type, files, class_label, dummy_var="truthProng", cuts=None):
        """
        Class constructor - fills in meta-data for the data type
        :param data_type: The type of data file being loaded e.g. Gammatautau, JZ1, ect...
        :param files: A list of files of the same data type to be loaded
        :param class_label: 1 for signal, 0 for background
        :param dummy_var: A variable to be loaded from the file to be loaded and iterated through to work out the number
        of events in the data files
        """
        self.data_type = data_type
        self.files = files
        self.dummy_var = dummy_var
        self.cut = cuts
        dummy_array = uproot.concatenate(files, filter_name="TauJets."+dummy_var, cut=cuts)
        self._num_events = len(dummy_array["TauJets."+dummy_var])

        self.specific_batch_size = 0
        self.batches = None
        self.class_label = class_label
        self._n_events_in_batch = 0
        self._batches_generator = None
        self._current_index = 0

        logger.log(f"Found {len(files)} files for {data_type}", 'INFO')
        logger.log(f"Found these files: {files}", 'INFO')
        logger.log(f"Found {self._num_events} events for {data_type}", 'INFO')
        logger.log(f"DataLoader for {data_type} initialized", "INFO")

    def number_of_batches(self, batch_size, nevents_total):
        counter = 0
        self.specific_batch_size = math.ceil((self._num_events / nevents_total) * batch_size)

        for _ in uproot.iterate(self.files, filter_name="TauJets." + self.dummy_var, step_size=self.specific_batch_size,
                                library="ak", cut=self.cut):
            counter += 1
        return counter

    def num_events(self):
        return self._num_events

    def set_batches(self,  variable_list, total_num_events, total_batch_size):
        self.specific_batch_size = self._num_events // total_batch_size
        self._batches_generator = uproot.iterate(self.files, filter_name=variable_list, step_size=self.specific_batch_size,
                                       library="ak", cut=self.cut)

    def get_next_batch(self):
        batch = next(self._batches_generator)
        logger.log(f"{self.data_type}: Loaded batch {self._current_index}", 'DEBUG')
        self._current_index += 1
        return batch, np.ones(len(batch)) * self.class_label



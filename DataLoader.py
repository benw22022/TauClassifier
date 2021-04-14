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

class DataLoader:

    def __init__(self, data_type, files, class_label, dummy_var="mcEventWeight", cut=None):
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
        self.cut = cut

        dummy_array = uproot.concatenate(files, filter_name="TauJets."+dummy_var, cut=cut)
        self.num_events = len(dummy_array["TauJets."+dummy_var])

        self.specific_batch_size = 0
        self.batches = None
        self.class_label = class_label
        self._n_events_in_batch = 0
        logger.log(f"Found {len(files)} files for {data_type}", 'INFO')
        logger.log(f"Found these files: {files}", 'INFO')
        logger.log(f"Found {self.num_events} events for {data_type}", 'INFO')
        logger.log(f"DataLoader for {data_type} initialized", "INFO")

    def load_batches(self, variable_list, total_num_events, total_batch_size):
        """
        Chunks data up into batches of a specific size and lazily loads them. The iterate step_size is chosen so that
        there is a proportional number of each type of event in each batch of training data. This is done separately,
        outside of the constructor, since we need to know the total number of events in all files in order to properly
        weight the step_size
        NOTE: Is actually very memory intensive/slow
        :param variable_list: List of variables to be loaded
        :param total_num_events: Total number of events in all data files
        :param total_batch_size: Size of batch to be trained on
        :return: None
        """
        self.specific_batch_size = math.ceil(total_batch_size * self.num_events / total_num_events)
        self.batches = [batch for batch in uproot.iterate(self.files, filter_name=variable_list, step_size=self.specific_batch_size,
                                                          library="ak", cut=self.cut)]
        #how = "zip",
        logger.log(f"Preloaded data for {self.data_type}", 'INFO')

    def number_of_batches(self, total_num_events, total_batch_size):
        counter = 0
        self.specific_batch_size = math.ceil(total_batch_size * self.num_events / total_num_events)
        for _ in uproot.iterate(self.files, filter_name="TauJets." + self.dummy_var, step_size=self.specific_batch_size,
                                library="ak", how="zip"):
            counter += 1
        return counter


    def load_batch(self, idx, variable_list, total_num_events, total_batch_size):

        self.specific_batch_size = math.ceil(total_batch_size * self.num_events / total_num_events)

        logger.log(f"Loading array {idx} for {self.data_type}...", 'DEBUG')

        data = uproot.lazy(self.files, filter_name=variable_list, step_size=self.specific_batch_size)
        batch = data[idx * self.specific_batch_size: idx * self.specific_batch_size + self.specific_batch_size]

        logger.log(f"Loaded array {idx} for {self.data_type} ", 'DEBUG')

        if self.data_type == "Gammatautau":
            batch = batch[batch["TauJets.truthProng"] == 1]

        return batch, np.ones(len(batch)) * self.class_label



"""
DataGenerator Class Definition
________________________________________________________________________________________________________________________
Class that is used to feed Keras batches of data for training/testing/validation so that we don't have to load all the
data into memory at once
TODO: Make this generalisable to different problems
"""

import numpy as np
import keras
from DataLoader import DataLoader, apply_scaling
from utils import logger, find_anomalous_entries
import tensorflow as tf
import datetime
import time
import gc
import ray
from ray.util import inspect_serializability


class DataGenerator(keras.utils.Sequence):

    def __init__(self, file_handler_list, variables_dict, nbatches=1000, cuts=None, label="DataGenerator",
                 _benchmark=False, extra_return_var=None):
        """
        Class constructor - loads batches of data in a way that can be fed one by one to Keras - avoids having to load
        entire dataset into memory prior to training
        :param file_dict: A dictionary with keys labeling the data type (e.g. Gammatautau, JZ1, etc...) and values being
        a list of files corresponding to that data type
        :param variables_dict: A dictionary of input variables with keys labeling the variable type (Tracks, Clusters, etc...)
        and values being a list of branch names of the variables associated with that type
        :param nbatches: Number of batches to *roughly* split the dataset into (not exact due to uproot inconstantly
        changing batch size when moving from one file to another)
        :param cuts: A dictionary of cuts with keys the same as file_dict. The values should be a string which can be
        parsed by uproot's cut option e.g. "(pt1 > 50) & ((E1>100) | (E1<90))"
        :param label: A string to label the generator with - useful when debugging multiple generators
        :param extra_return_var: An additional array to return when making a batch - use for producing
        """
        logger.log(f"Initializing DataGenerator: {label}", 'INFO')
        self._file_handlers = file_handler_list
        self.label = label
        self.data_loaders = []
        self.cuts = cuts
        self._current_index = 0
        self._epochs = 0
        self.__benchmark = _benchmark
        self.extra_return_var = extra_return_var

        logger.log(f"self.extra_return_var = {self.extra_return_var}")

        # Organise a list of all variables
        self._variables_dict = variables_dict
        self._variables_list = []
        for _, variable_list in variables_dict.items():
            self._variables_list += variable_list

        for file_handler in self._file_handlers:
            if cuts is not None and file_handler.label in cuts:
                logger.log(f"Cuts applied to {file_handler.label}: {self.cuts[file_handler.label]}")

                dl_label = file_handler.label + "_" + self.label
                file_list = file_handler.file_list
                class_label = file_handler.class_label

                # Check that dataloader is serializable - needs to be for ray to work
                inspect_serializability(DataLoader, name="test")
                dl = DataLoader.remote(file_handler.label, file_list, class_label, nbatches, variables_dict, cuts=self.cuts[file_handler.label],
                                                    label=label, extra_return_var=extra_return_var)
                self.data_loaders.append(dl)

            else:
                dl_label = file_handler.label + "_" + self.label
                file_list = file_handler.file_list
                class_label = file_handler.class_label
                logger.log(inspect_serializability(DataLoader, name="test"))
                dl = DataLoader.remote(file_handler.label, file_list, class_label, nbatches, variables_dict, label=dl_label, extra_return_var=extra_return_var)
                self.data_loaders.append(dl)


        # Get number of events in each dataset
        self._total_num_events = []
        num_batches_list = []
        for data_loader in self.data_loaders:
            self._total_num_events.append(ray.get(data_loader.num_events.remote()))
            num_batches_list.append(ray.get(data_loader.number_of_batches.remote()))
        logger.log(f"{self.label} - Found {sum(self._total_num_events)} events total", "INFO")

        # Work out how many batches to split the data into
        self._num_batches = min(num_batches_list)

    def load_batch(self):
        """
        Loads a batch of data from each data type and concatenates them into single arrays for training
        :param idx: The index of the batch of data to retrieve
        :return: A list of arrays to be passed to model.fit()
        """

        batch_load_time = time.time()
        extra_arr = []

        # Concatenate the results from each file stream together
        if self.extra_return_var is not None:
            res = ray.get([dl.set_batch.remote() for dl in self.data_loaders])
            logger.log(f"res = {res} - len = {len(res)}")
            batch = res[0][0]
            extra_arr = res[0][1]
            try:
                extra_arr = np.concatenate([arr for arr in extra_arr])
            except ValueError:
                pass

        else:
            batch = ray.get([dl.get_batch.remote() for dl in self.data_loaders])

        track_array = np.concatenate([result[0][0] for result in batch])
        neutral_pfo_array = np.concatenate([result[0][1] for result in batch])
        shot_pfo_array = np.concatenate([result[0][2] for result in batch])
        conv_track_array = np.concatenate([result[0][3] for result in batch])
        jet_array = np.concatenate([result[0][4] for result in batch])
        label_array = np.concatenate([result[1] for result in batch])
        weight_array = np.concatenate([result[2] for result in batch])

        load_time = str(datetime.timedelta(seconds=time.time()-batch_load_time))
        logger.log(f"{self.label}: Processed batch {self._current_index}/{self.__len__()} - {len(label_array)} events"
                   f" in {load_time}", "INFO")

        if self.__benchmark:
            return ((track_array, neutral_pfo_array, shot_pfo_array, conv_track_array, jet_array), label_array, weight_array),\
                    self._current_index, len(label_array), load_time


        try:
            if self.extra_return_var is not None:
                return (track_array, neutral_pfo_array, shot_pfo_array, conv_track_array,
                        jet_array), label_array, weight_array, extra_arr
            return (track_array, neutral_pfo_array, shot_pfo_array, conv_track_array, jet_array), label_array, weight_array
        finally:
            del batch
            del track_array, neutral_pfo_array, shot_pfo_array, conv_track_array, jet_array, label_array, weight_array
            gc.collect()

    def predict(self, model):
        y_pred = []
        y_true = []
        weights = []
        for i in range(0, len(self)):
            batch_tmp, y_true_tmp, weights_tmp = self[i]
            y_pred_tmp = model.predict(batch_tmp)
            y_pred.append(y_pred_tmp)
            y_true.append(y_true_tmp)
            weights.append(weights_tmp)
        self.reset_generator()
        y_true = np.concatenate([arr for arr in y_true])
        y_pred = np.concatenate([arr for arr in y_pred])
        weights = np.concatenate([arr for arr in weights])
        return y_pred, y_true, weights

    def __len__(self):
        """
        This returns the number of batches that the data was split up into (important that this is correct or you'll
        run into memory access violations when Keras oversteps bounds of array)
        :return: The number of batches in an epoch
        """
        return self._num_batches

    def __getitem__(self, idx):
        """
        Overloads [] operator - allows generator to be indexable. This must be provided so that Keras can use generator
        Kinda hacky - ideally since this is generator we would be using the __next__ function - but Keras wants
        indexable data - so __getitem__ it is.
        :param idx: An index - doesn't actually get used for anything - you can ignore it
        :return: The next batch of data
        """
        self._current_index += 1
        try:
            return self.load_batch()
        finally:
            # If we reach the end of the generator we reset so we can loop again
            if self._current_index == len(self):
                self.reset_generator()

    def __next__(self):
        """
        Overloads next() operator - allows generator to be iterable
        - This must be defined for DataGenerator to be converted to a TensorFlow dataset (if that is something you want
        to do)
        :return: The next batch of data
        """

        if self._current_index < self.__len__():
            self._current_index += 1
            return self.load_batch()
        raise StopIteration

    def __iter__(self):
        return self

    def __call__(self):
        return self

    def reset_generator(self):
        """
        Function that will reset the generator and reset the dataloader objects
        index will also get reset to zero
        :return:
        """
        self._current_index = 0
        for data_loader in self.data_loaders:
            data_loader.reset_dataloader.remote()

    def on_epoch_end(self):
        """
        This function is called by Keras at the end of every epoch. Here it is used to reset the iterators to the start
        :return:
        """
        self.reset_generator()

    def number_events(self):
        return self._total_num_events

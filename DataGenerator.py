"""
DataGenerator Class Definition
________________________________________________________________________________________________________________________
Class that is used to feed Keras batches of data for training/testing/validation so that we don't have to load all the
data into memory at once
TODO: Make this generalisable to different problems
"""

import numpy as np
import keras
from DataLoaderMK2 import DataLoader
from utils import logger
import tensorflow as tf
import datetime
import time
import gc
import ray
from ray.util import inspect_serializability



def process_wrapper(dl, idx):
    dl.set_batch(idx)


class DataGenerator(keras.utils.Sequence):

    def __init__(self, file_handler_list, variables_dict, nbatches=1000, epochs=50, cuts=None, label="DataGenerator",
                 _benchmark=False):
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
        """
        logger.log(f"Initializing DataGenerator: {label}", 'INFO')
        self._file_handlers = file_handler_list
        self.label = label
        self.data_loaders = []
        self.cuts = cuts
        self._current_index = 0
        self._epochs = 0
        self.__benchmark = _benchmark



        # Organise a list of all variables
        self._variables_dict = variables_dict
        self._variables_list = []
        for _, variable_list in variables_dict.items():
            self._variables_list += variable_list

        # Managers
        # self.manager = BaseManager()
        # self.manager.register('DataLoader', DataLoader)
        # self.manager.start()
        # self.inst = []
        #ray.init(object_store_memory=1e9, _memory=1e9)
        #ray.init(object_store_memory=1e9)

        for file_handler in self._file_handlers:
            if cuts is not None and file_handler.label in cuts:
                logger.log(f"Cuts applied to {file_handler.label}: {self.cuts[file_handler.label]}")

                label = file_handler.label
                file_list = file_handler.file_list
                class_label = file_handler.class_label

                # Check that dataloader is serializable - needs to be for ray to work
                inspect_serializability(DataLoader, name="test")
                dl = DataLoader.remote(label, file_list, class_label, nbatches, variables_dict, cuts=self.cuts[label],
                                                    label=label)
                self.data_loaders.append(dl)

            else:
                label = file_handler.label
                file_list = file_handler.file_list
                class_label = file_handler.class_label
                logger.log(inspect_serializability(DataLoader, name="test"))
                dl = DataLoader.remote(label, file_list, class_label, nbatches, variables_dict,
                                                    label=label)
                self.data_loaders.append(dl)


        # Get number of events in each dataset
        self._total_num_events = []
        num_batches_list = []
        for data_loader in self.data_loaders:
            self._total_num_events.append(data_loader.num_events.remote())
            num_batches_list.append(ray.get(data_loader.number_of_batches.remote()))
        #logger.log(f"Found {sum(self._total_num_events)} events total", "INFO")

        # Work out how many batches to split the data into
        self._num_batches = min(num_batches_list)

        # # Lazily load batches of data for each dataset
        # for data_loader in self.data_loaders:
        #     logger.log(f"Number of batches in {data_loader.data_type()} = {data_loader.number_of_batches()}")
        #
        # # Work out the number of batches for training epoch (important)
        # logger.log(f"Number of batches per epoch: {self._num_batches}", 'DEBUG')
        # logger.log("DataGenerator initialized", 'INFO')




    def load_batch(self, idx=None):
        """
        Loads a batch of data from each data type and concatenates them into single arrays for training
        :param idx: The index of the batch of data to retrieve
        :return: A list of arrays to be passed to model.fit()
        """

        batch_load_time = time.time()

        index = idx
        if idx is None:
            index = self._current_index

        batch = None
        multiprocess = True

        if not multiprocess:
            batch = [dl.load_batch_from_data(idx=index) for dl in self.data_loaders]
        else:

            [dl.set_batch.remote(idx) for dl in self.data_loaders]
            futures = [dl.get.remote() for dl in self.data_loaders]
            batch = ray.get(futures)

        logger.log(f"Loaded batch {self.label} {self._current_index}/{self.__len__()} in {str(datetime.timedelta(seconds=time.time()-batch_load_time))}", "DEBUG")

        track_array = np.concatenate([sub_batch[0][0] for sub_batch in batch])
        neutral_pfo_array = np.concatenate([sub_batch[0][1] for sub_batch in batch])
        shot_pfo_array = np.concatenate([sub_batch[0][2] for sub_batch in batch])
        conv_track_array = np.concatenate([sub_batch[0][3] for sub_batch in batch])
        jet_array = np.concatenate([sub_batch[0][4] for sub_batch in batch])
        label_array = np.concatenate([sub_batch[1] for sub_batch in batch])
        weight_array = np.concatenate([sub_batch[2] for sub_batch in batch])

        #find_anomalous_entries(track_array, 20, logger, arr_name="tracks")
        #find_anomalous_entries(neutral_pfo_array, 20, logger, arr_name="neutral PFO")
        #find_anomalous_entries(shot_pfo_array, 20, logger, arr_name="shot PFO")
        #find_anomalous_entries(conv_track_array, 20, logger, arr_name="conv track")
        #find_anomalous_entries(jet_array, 20, logger, arr_name="jets")
        #find_anomalous_entries(weight_array, 5, logger, arr_name="weights")

        logger.log(f"Batch: {self._current_index}/{self.__len__()} - shapes:", 'DEBUG')
        logger.log(f"TauTracks Shape = {track_array.shape}", 'DEBUG')
        logger.log(f"ConvTracks Shape = {conv_track_array.shape}", 'DEBUG')
        logger.log(f"ShotPFO Shape = {shot_pfo_array.shape}", 'DEBUG')
        logger.log(f"NeutralPFO Shape = {neutral_pfo_array.shape}", 'DEBUG')
        logger.log(f"TauJets Shape = {jet_array.shape}", 'DEBUG')
        logger.log(f"Labels Shape = {label_array.shape}", 'DEBUG')
        logger.log(f"Weight Shape = {weight_array.shape}", 'DEBUG')

        start_time = time.time()
        track_array = tf.convert_to_tensor(track_array.astype("float32"))
        conv_track_array = tf.convert_to_tensor(conv_track_array.astype("float32"))
        shot_pfo_array = tf.convert_to_tensor(shot_pfo_array.astype("float32"))
        neutral_pfo_array = tf.convert_to_tensor(neutral_pfo_array.astype("float32"))
        jet_array = tf.convert_to_tensor(jet_array.astype("float32"))
        label_array = tf.convert_to_tensor(label_array.astype("float32"))
        weight_array = tf.convert_to_tensor(weight_array.astype("float32"))
        logger.log(f"Tensor Conversion time = {time.time() - start_time} seconds", level='DEBUG')


        load_time = str(datetime.timedelta(seconds=time.time()-batch_load_time))
        logger.log(f"{self.label}: Processed batch {self._current_index}/{self.__len__()} - {len(label_array)} events"
                   f" in {load_time}", "INFO")

        # Make sure we clear up after ourselves
        for elem in batch:
            del elem
        del batch
        for elem in futures:
            del elem
        del futures
        gc.collect()

        if self.__benchmark:
            return ((track_array, neutral_pfo_array, shot_pfo_array, conv_track_array, jet_array), label_array, weight_array),\
                    self._current_index, len(label_array), load_time


        return (track_array, neutral_pfo_array, shot_pfo_array, conv_track_array, jet_array), label_array, weight_array

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
        :param idx: An index
        :return: A full batch of data
        """
        logger.log(f"{self.label} __getitem__ called with index = {idx}", 'DEBUG')
        self._current_index += 1
        try:
            return self.load_batch(idx=self._current_index)
        finally:
            # Clear Memory of last batch before the end of the epoch -
            if self._current_index == len(self):
                self.reset_generator()

    def __next__(self):
        """
        Overloads next() operator - allows generator to be iterable - This is what will be called when
        tf.data.Dataset.from_generator() is used
        :return:
        """

        if self._current_index < self._num_batches:
            self._current_index += 1
            return self.load_batch()

        self._epochs += 1
        if self._epochs < self._num_batches * self.__len__():
            self._current_index = -1
            return self.__next__()

        #logger.log(f"{self.label} __next__ called")
        #if self._epoch_index < self._epochs:
        #    if self._current_index < self._num_batches:
        #        self._current_index += 1
        #        return self.load_batch()
        #    self.on_epoch_end()
        #    self._epoch_index += 1
        #    if self.label == "Validation Generator":
        #        logger.log(f"StopIteration raised by __next__ in {self.label}")
        #        raise StopIteration
        #    logger.log(f"Completed {self._epoch_index} epochs")
        #    return self.load_batch()
        raise StopIteration

    def __iter__(self):
        return self

    def __call__(self):
        self.on_epoch_end()
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
        gc.collect()

    def on_epoch_end(self):
        """
        This function is called by Keras at the end of every epoch. Here it is used to reset the iterators to the start
        :return:
        """
        self.reset_generator()

    def number_events(self):
        return self._total_num_events

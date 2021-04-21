"""
DataGenerator Class Definition
________________________________________________________________________________________________________________________
Class that is used to feed Keras batches of data for training/testing/validation so that we don't have to load all the
data into memory at once
TODO: Potential performance improvement could be found by loading up the next batch of training whilst the the previous
TODO: batch is being trained on
"""
import numpy as np
import keras
import awkward as ak
from DataLoader import DataLoader
from utils import logger
import time
import datetime
import tensorflow as tf
from preprocessing import finite_log
import time
import datetime
from multiprocessing import Pool
from functools import partial
import time
import threading
from multiprocessing.dummy import Pool  # dummy uses threads

N_WORKERS = 10


def multi_threaded(gens):
    combi_g = mt_gen(gens)
    results = []
    for item in combi_g:
        results.append(item)
    return results


def mt_gen(gens):
    with Pool(N_WORKERS) as pool:
        #while True:
        async_results = [pool.apply_async(next, args=(g,)) for g in gens]
        try:
            results = [r.get() for r in async_results]
        except StopIteration:  # needed for Python 3.7+, PEP 479, bpo-32670
            return
        yield results

class DataGenerator(keras.utils.Sequence):

    def __init__(self, file_dict, variables_dict, nbatches=1000, cuts=None):
        """
        Class constructor - loads batches of data in a way that can be fed one by one to Keras - avoids having to load
        entire dataset into memory prior to training
        :param file_dict: A dictionary with keys labeling the data type (e.g. Gammatautau, JZ1, etc...) and values being
        a list of files corresponding to that data type
        :param variables_dict: A dictionary of input variables with keys labeling the variable type (Tracks, Clusters, etc...)
        and values being a list of branch names of the variables associated with that type
        :param batch_size: The number of events to train on per batch of data
        """
        logger.log("Initializing DataGenerator", 'INFO')
        self._nbatches = 10000
        self._file_dict = file_dict
        self.data_loaders = []
        self.cuts = cuts
        self._current_index = 0

        self.nprocs = 12

        # Organise a list of all variables
        self._variables_dict = variables_dict
        self._variables_list = []
        for _, variable_list in variables_dict.items():
            self._variables_list += variable_list

        # Load each data type separately
        for data_type, file_list in file_dict.items():
            class_label = 0
            if data_type == "Gammatautau":
                class_label = 1
            if cuts is not None and data_type in cuts:
                logger.log(f"Cuts applied to {data_type}: {self.cuts[data_type]}")
                self.data_loaders.append(DataLoader(data_type, file_list, class_label, nbatches, variables_dict, cuts=self.cuts[data_type]))
            else:
                self.data_loaders.append(DataLoader(data_type, file_list, class_label, nbatches,  variables_dict))

        # Get number of events in each dataset
        self._total_num_events = []
        self._num_batches_list = []
        for data_loader in self.data_loaders:
            self._total_num_events.append(data_loader.num_events())
            self._num_batches_list.append(data_loader.number_of_batches())
        logger.log(f"Found {sum(self._total_num_events)} events total", "INFO")

        # Work out how many batches to split the data into
        self._num_batches = min(self._num_batches_list)

        # Lazily load batches of data for each dataset
        for data_loader in self.data_loaders:
            data_loader.set_batches(self._variables_list)
            logger.log(f"Number of batches in {data_loader.data_type()} = {data_loader.number_of_batches()}")

        # Work out the number of batches for training epoch (important)
        logger.log(f"Number of batches per epoch: {self._num_batches}", 'DEBUG')
        logger.log("DataGenerator initialized", 'INFO')


    def load_batch(self, idx):
        """
        Loads a batch of data from each data type and concatenates them into single arrays for training
        NOTE: idx argument is still being passed but is actually unused. This is for compatibility reasons - I would like
        to test uproot.lazy still in the wider tensorflow context so will keep idx for the time being
        :param idx: The index of the batch of data to retrieve
        :return: A list of arrays to be passed to model.fit()
            [[Tracks, Clusters, Jets], labels, weights]
        """

        batch_load_time = time.time()

        track_array = np.array([])
        conv_track_array = np.array([])
        shot_pfo_array = np.array([])
        neutral_pfo_array = np.array([])
        jet_array = np.array([])
        label_array = np.array([])
        weight_array = np.array([])

        for i in range(0, len(self.data_loaders)):
            if i == 0:
                track_array, conv_track_array, shot_pfo_array, neutral_pfo_array, jet_array, \
                label_array, weight_array = self.data_loaders[i].load_batch_from_data(idx)
                logger.log(f"Preparing batch: Done {self.data_loaders[i].data_type} {i+1}/{len(self.data_loaders)}", 'HELPME')
            else:
                tmp_track_array, tmp_conv_track_array, tmp_shot_pfo_array, tmp_neutral_pfo_array, tmp_jet_array, \
                tmp_label_array, tmp_weight_array =  self.data_loaders[i].load_batch_from_data(idx)
                track_array = np.concatenate((tmp_track_array, track_array))
                conv_track_array = np.concatenate((tmp_conv_track_array, conv_track_array))
                shot_pfo_array = np.concatenate((tmp_shot_pfo_array, shot_pfo_array))
                neutral_pfo_array = np.concatenate((tmp_neutral_pfo_array, neutral_pfo_array))
                jet_array = np.concatenate((tmp_jet_array, jet_array))
                label_array = np.concatenate((tmp_label_array, label_array))
                weight_array = np.concatenate((tmp_weight_array, weight_array))
                logger.log(f"Preparing batch: Done {self.data_loaders[i].data_type} {i+1}/{len(self.data_loaders)}", 'HELPME')

        logger.log(f"Loaded batch {self._current_index}/{self.__len__()} in {str(datetime.timedelta(seconds=time.time()-batch_load_time))}", "DEBUG")
        self._current_index += 1

        track_array = tf.convert_to_tensor(track_array.astype("float32"))
        conv_track_array = tf.convert_to_tensor(conv_track_array.astype("float32"))
        shot_pfo_array = tf.convert_to_tensor(shot_pfo_array.astype("float32"))
        neutral_pfo_array = tf.convert_to_tensor(neutral_pfo_array.astype("float32"))
        jet_array = tf.convert_to_tensor(jet_array.astype("float32"))
        label_array = tf.convert_to_tensor(label_array.astype("float32"))
        weight_array = tf.convert_to_tensor(weight_array.astype("float32"))

        logger.log(f"Batch: {self._current_index}/{self.__len__()} - shapes:", 'DEBUG')
        logger.log(f"TauTracks Shape = {track_array.shape}", 'DEBUG')
        logger.log(f"ConvTracks Shape = {conv_track_array.shape}", 'DEBUG')
        logger.log(f"ShotPFO Shape = {shot_pfo_array.shape}", 'DEBUG')
        logger.log(f"NeutralPFO Shape = {neutral_pfo_array.shape}", 'DEBUG')
        logger.log(f"TauJets Shape = {jet_array.shape}", 'DEBUG')
        logger.log(f"Labels Shape = {label_array.shape}", 'DEBUG')
        logger.log(f"Weight Shape = {weight_array.shape}", 'DEBUG')

        return [track_array, neutral_pfo_array, shot_pfo_array, conv_track_array, jet_array], label_array, weight_array


    def multithreaded_load(self, idx):

        gens = []
        for dataloader in self.data_loaders:
            gens.append(dataloader.load_batch_from_data(idx))
        results = np.array(multi_threaded(gens), dtype='object')

        track_array = np.array([], dtype="float32")
        conv_track_array = np.array([], dtype="float32")
        shot_pfo_array = np.array([], dtype="float32")
        neutral_pfo_array = np.array([], dtype="float32")
        jet_array = np.array([], dtype="float32")
        label_array = np.array([], dtype="float32")
        weight_array = np.array([], dtype="float32")

        for i in range(0, results.shape[1]):
            if i == 0:
                track_array, conv_track_array, shot_pfo_array, neutral_pfo_array, jet_array, \
                label_array, weight_array = results[0][i]
            else:
                tmp_track_array, tmp_conv_track_array, tmp_shot_pfo_array, tmp_neutral_pfo_array, tmp_jet_array, \
                tmp_label_array, tmp_weight_array = results[0][i]
                track_array = np.concatenate((tmp_track_array, track_array))
                conv_track_array = np.concatenate((tmp_conv_track_array, conv_track_array))
                shot_pfo_array = np.concatenate((tmp_shot_pfo_array, shot_pfo_array))
                neutral_pfo_array = np.concatenate((tmp_neutral_pfo_array, neutral_pfo_array))
                jet_array = np.concatenate((tmp_jet_array, jet_array))
                label_array = np.concatenate((tmp_label_array, label_array))
                weight_array = np.concatenate((tmp_weight_array, weight_array))


        track_array = tf.convert_to_tensor(track_array.astype("float32"))
        conv_track_array = tf.convert_to_tensor(conv_track_array.astype("float32"))
        shot_pfo_array = tf.convert_to_tensor(shot_pfo_array.astype("float32"))
        neutral_pfo_array = tf.convert_to_tensor(neutral_pfo_array.astype("float32"))
        jet_array = tf.convert_to_tensor(jet_array.astype("float32"))
        label_array = tf.convert_to_tensor(label_array.astype("float32"))
        weight_array = tf.convert_to_tensor(weight_array.astype("float32"))

        """
        logger.log(f"Batch: {self._current_index}/{self.__len__()} - shapes:", 'DEBUG')
        logger.log(f"TauTracks Shape = {track_array.shape}", 'DEBUG')
        logger.log(f"ConvTracks Shape = {conv_track_array.shape}", 'DEBUG')
        logger.log(f"ShotPFO Shape = {shot_pfo_array.shape}", 'DEBUG')
        logger.log(f"NeutralPFO Shape = {neutral_pfo_array.shape}", 'DEBUG')
        logger.log(f"TauJets Shape = {jet_array.shape}", 'DEBUG')
        logger.log(f"Labels Shape = {label_array.shape}", 'DEBUG')
        logger.log(f"Weight Shape = {weight_array.shape}", 'DEBUG')
        """
        return [track_array, neutral_pfo_array, shot_pfo_array, conv_track_array, jet_array], label_array, weight_array

    def get_batch_shapes(self):
        """
        Loads a batch at a specific index and returns the shapes of the returned arrays
        Used for debugging and initializing network input layers
        :param idx: An index (default is zero)
        :return: A list of all the array shapes.
            tracks.shape, clusters.shape, jets.shape, labels.shape, weights.shape
        """

        #batch = self.load_batch(0)
        batch = self.multithreaded_load(0)
        shapes = []

        for item in batch[0]:
            shapes.append(item.shape)
        shapes.append(batch[1].shape)
        shapes.append(batch[2].shape)

        return shapes


    def __len__(self):
        """
        This returns the number of batches that the data was split up into (important that this is correct or you'll
        run into memory access violations when Keras oversteps bounds of array)
        :return: The number of batches in an epoch
        """
        return self._num_batches - 1

    def __getitem__(self, idx):
        """
        The function that Keras will call in order to get a new batch of data to train on. Same as self.load_batch().
        :param idx: An index - set by Keras
        :return: A full batch of data
        """
        logger.log(f"loaded batch {idx}/{self.__len__()}", 'DEBUG')

        #return self.load_batch(idx)
        return self.multithreaded_load(idx)

    def reset_generator(self):
        """
        Function that will reset the generator by recreating the uproot.iterator object for each dataloader
        index will also get reset to zero
        :return:
        """
        self._current_index = 0
        for data_loader in self.data_loaders:
            data_loader.set_batches(self._variables_list)

    def on_epoch_end(self):
        """
        This function is called by Keras at the end of every epoch. Here it is used to reset the iterators to the start
        :return:
        """
        logger.log_memory_usage(level='INFO')
        self.reset_generator()

    def set_max_itr(self, max_itr):
        """
        A function to set the max number of iterations to go through so that we don't run into errors. I really need to
        find a better solution to this.
        :param max_itr:
        :return:
        """
        self._num_batches = max_itr
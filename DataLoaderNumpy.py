"""
DataLoader Class definition
________________________________________________________________________________________________________________________
A helper class and ray actor to generate batches of data from a set of root files. This object applies all the required
transformations to the data and computes the class labels.
"""

import math
import awkward as ak
import tensorflow as tf
import numpy as np
import uproot
from preprocessing import reweighter
from utils import logger
import ray
import gc
import time
import numba as nb
from utils import Result
from sklearn.preprocessing import StandardScaler

#@nb.njit()
def apply_scaling(np_arrays, dummy_val=0, thresh=45, flag=False):
    """
    Rescales each varaible to be between zero and one. Function is jitted for speed
    :param np_arrays: The numpy arrays containing a set of input variables
    :return: A new array containing the rescaled data
    """
    print(np_arrays.shape)

    # if len(np_arrays.shape) == 3:
    i = 0
    while True:
        try:
            arr = np.ravel(np_arrays[:, i])
            #arr = arr[arr != dummy_val]
            arr_median = np.median(arr)
            q75, q25 = np.percentile(arr, [75, 25])
            arr_iqr = q75 - q25

            if arr_iqr != 0:
                np_arrays[:, i] = (np_arrays[:, i] - arr_median) / arr_iqr #np.where(np_arrays[:, i] != dummy_val, (np_arrays[:, i] - arr_median) / arr_iqr, dummy_val)
                np_arrays[:, i] = np.where(np_arrays[:, i] < thresh, np_arrays[:, i], thresh)
                np_arrays[:, i] = np.where(np_arrays[:, i] > -thresh, np_arrays[:, i], -thresh)

            i += 1
            print(f"i = {i}")
        except:
            print(f"ERROR at i = {i}")
        return np_arrays

@nb.njit()
def labeler(truth_decay_mode_np_array, labels_np_array):
    """
    Function to compute decay mode labels for Gammatautau. Due to large for loop, the function is jitted for speed
    :param truth_decay_mode_np_array: The Truth Decay Mode - number of neutral hadrons in decay
    :param labels_np_array: The array of labels - already initialized ready to be modified
    Convention:
    [1, 0, 0, 0] == Background Jets
    [0, 1, 0, 0] == 1p0n
    [0, 0, 1, 0] == 1p1n
    [0, 0, 0, 1] == 1pXn
    :return: The correctly labelled array
    """
    for i in range(0, len(truth_decay_mode_np_array, )):
        elem = truth_decay_mode_np_array[i]
        if elem == 0:
            labels_np_array[i][1] = 1
        elif elem == 1:
            labels_np_array[i][2] = 1
        else:
            labels_np_array[i][3] = 1
    return labels_np_array



@ray.remote
class DataLoader:

    def __init__(self, data_type, files, class_label, nbatches, variables_dict, dummy_var="truthProng", cuts=None,
                 batch_size=None, label="Dataloader"):
        """
        Class constructor - fills in meta-data for the data type
        :param data_type: The type of data file being loaded e.g. Gammatautau, JZ1, ect...
        :param files: A list of files of the same data type to be loaded
        :param class_label: 1 for signal, 0 for background
        :param variables_dict: dictionary of variables to load
        :param nbatches: number of batches to *roughly* split the data into
        :param dummy_var: A variable to be loaded from the file to be loaded and iterated through to work out the number
        of events in the data files.
        :param cuts: A string which can be parsed by uproot's cut option e.g. "(pt1 > 50) & ((E1>100) | (E1<90))"
        :param batch_size: Allows you to manually set the batch size for the data. This will override the automatically
        calculated batch size inferred from nbatches
        """
        self._data_type = data_type
        self.label = label
        self.files = files
        self.dummy_var = dummy_var
        self.cut = cuts
        self._nbatches = nbatches
        self.class_label = class_label
        self._variables_dict = variables_dict
        self._current_index = 0

        # Parse variables
        self._variables_list = []
        for _, variable_list in variables_dict.items():
            self._variables_list += variable_list

        # Work out how many events there in the sample
        test_arr = uproot.concatenate(self.files, filter_name="TauJets." + self.dummy_var, step_size=10000,
                                      cut=self.cut, library='np')
        self._num_events = len(test_arr["TauJets." + self.dummy_var])

        # Set the DataLoader's batch size
        if batch_size is None:
            self.specific_batch_size = math.ceil(self._num_events / nbatches)
        else:
            self.specific_batch_size = batch_size

        # Setup the iterator
        self._batches_generator = uproot.iterate(self.files, filter_name=self._variables_list, cut=self.cut,
                                                 step_size=self.specific_batch_size, library='np')

        # Work out the number of batches there are in the generator
        self._num_real_batches = 0
        for _ in uproot.iterate(self.files, filter_name=self._variables_list[0], cut=self.cut,
                                step_size=self.specific_batch_size):
            self._num_real_batches += 1

        logger.log(f"Found {len(files)} files for {data_type}", 'INFO')
        logger.log(f"Found these files: {files}", 'INFO')
        logger.log(f"Found {self._num_events} events for {data_type}", 'INFO')
        logger.log(f"Number of batches in {self.label} {self.data_type()} = {self._num_real_batches}", 'DEBUG')
        logger.log(f"DataLoader for {data_type} initialized", "INFO")

    def next_batch(self):
        try:
            batch = next(self._batches_generator)
        except StopIteration:
            self._batches_generator = uproot.iterate(self.files, filter_name=self._variables_list, cut=self.cut,
                                                     step_size=self.specific_batch_size, library='np')
            return self.next_batch()
        self._current_index += 1
        return batch, np.ones(len(batch)) * self.class_label

    def pad_and_reshape_nested_arrays(self, batch, variable_type, max_items=10):
        """
        Function that acts on nested data to read relevant variables, pad, reshape and convert data from uproot into
        rectilinear numpy arrays
        :param batch: A dict of awkward arrays from uproot
        :param variable_type: Variable type to be selected e.g. Tracks, Neutral PFO, Jets etc...
        :param max_items: Maximum number of tracks/PFOs etc... to be associated to event
        :return: a rectilinear numpy array of shape:
                (num events in batch, number of variables belonging to variable type, max_items)
        """
        variables = self._variables_dict[variable_type]
        # np_arrays = np.zeros((len(batch[variables[0]]), len(variables), None), dtype='object')
        # dummy_val = 0
        # thresh = 45
        # for i in range(0, len(variables)):
        #     np_arr = batch[variables[i]]
        #     np_arrays[:, i] = np_arr
        # np_arrays = np.nan_to_num(np_arrays, posinf=0, neginf=0, copy=False).astype("float64")

        arrays = []
        for var in variables:
            arr = batch[var]
            arr_median = np.median(arr)
            q75, q25 = np.percentile(arr, [75, 25])
            arr_iqr = q75 - q25
            if arr_iqr != 0:
                arr = (arr - arr_median) / arr_iqr
            arrays.append(tf.ragged.constant(arr))
        tensor = tf.ragged.stack(arrays)
        return tensor

    def reshape_arrays(self, batch, variable_type):
        """
        Function that acts on flat data to read relevant variables, reshape and convert data from uproot into
        rectilinear numpy arrays
        :param batch: A dict of awkward arrays from uproot
        :param variable_type: Variable type to be selected e.g. Tracks, Neutral PFO, Jets etc...
        :return: a rectilinear numpy array of shape:
                (num events in batch, number of variables belonging to variable type)
        """
        variables = self._variables_dict[variable_type]
        np_arrays = np.zeros((len(batch[variables[0]]), len(variables)))
        for i in range(0, len(variables)):
            np_arrays[:, i] = batch[variables[i]]
        np_arrays = np.nan_to_num(np_arrays, posinf=0, neginf=0, copy=False).astype("float64")
        return tf.ragged.constant(np_arrays)


    def set_batch(self, idx):
        """
        Loads a batch of data of a specific data type and then stores it for later retrieval.
        Pads ragged track and PFO arrays to make them rectilinear
        and reshapes arrays into correct shape for training. The clip option in ak.pad_none will truncate/extend each
        array so that they are all of a specific length- here we limit nested arrays to 20 items
        :param idx: The index of the batch to be processed
        """
        batch, sig_bkg_labels_np_array = self.next_batch()

        track_np_arrays = self.pad_and_reshape_nested_arrays(batch, "TauTracks", max_items=20)
        conv_track_np_arrays = self.pad_and_reshape_nested_arrays(batch, "ConvTrack", max_items=20)
        shot_pfo_np_arrays = self.pad_and_reshape_nested_arrays(batch, "ShotPFO", max_items=20)
        neutral_pfo_np_arrays = self.pad_and_reshape_nested_arrays(batch, "NeutralPFO", max_items=20)
        jet_np_arrays = self.reshape_arrays(batch, "TauJets")

        # Compute labels
        labels_np_array = np.zeros((len(sig_bkg_labels_np_array), 4))
        if sig_bkg_labels_np_array[0] == 0:
            labels_np_array[:, 0] = 1
        else:
            truth_decay_mode_np_array = batch["TauJets.truthDecayMode"].astype("int32")
            labels_np_array = labeler(truth_decay_mode_np_array, labels_np_array)

        # Apply pT re-weighting
        weight_np_array = np.ones(len(labels_np_array))
        if self.class_label == 0:
            bkg_pt = batch["TauJets.ptIntermediateAxis"]
            weight_np_array = reweighter.reweight(bkg_pt)

        result = Result(track_np_arrays, neutral_pfo_np_arrays, shot_pfo_np_arrays, conv_track_np_arrays,
                        jet_np_arrays, labels_np_array, weight_np_array)

        return result

    def reset_dataloader(self):
        self._current_index = 0
        self._batches_generator = uproot.iterate(self.files, filter_name=self._variables_list, cut=self.cut,
                                                 library='np', step_size=self.specific_batch_size)
        gc.collect()

    def num_events(self):
        return self._num_events

    def data_type(self):
        return self._data_type

    def number_of_batches(self):
        return self._num_real_batches

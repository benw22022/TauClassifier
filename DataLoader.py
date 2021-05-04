"""
DataLoader Class definition
________________________________________________________________________________________________________________________
A helper class to lazily load batches of training data for each type of data
"""

import math

import awkward as ak
import numpy as np
import uproot
from preprocessing import finite_log, pt_reweight
from utils import logger
from variables import log_list


def apply_logs(np_arrays, variables, log_list):
    #for i in range(0, len(variables)):
    #    if variables[i] in log_list:
    # Take logs of everything where the mean of the column is > 1
    for i in range(0, np_arrays.shape[1]):
        if np.mean(np_arrays[:, i]) > 1:
            np_arrays[:, i] = np.log(np_arrays[:, i], out=np.zeros_like(np_arrays[:, i]), where=(np_arrays[:, i] > 0))
    return np_arrays



class DataLoader:

    def __init__(self, data_type, files, class_label, nbatches, variables_dict, dummy_var="truthProng", cuts=None, batch_size=None,
                 label="Dataloader"):
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

        # If we run out of data in this data loader
        self._disable_indexing = False

        # Calculate the number of events in sample- Should all fit into memory so can use uproot.concatenate for speed
        dummy_array = uproot.concatenate(files, filter_name="TauJets."+dummy_var, cut=cuts)
        self._num_events = len(dummy_array["TauJets."+dummy_var])

        # Set the DataLoader's batch size
        if batch_size is None:
            self.specific_batch_size = math.ceil(self._num_events / nbatches)
        else:
            self.specific_batch_size = batch_size

        # Parse variables
        self._variables_list = []
        for _, variable_list in variables_dict.items():
            self._variables_list += variable_list

        # Setup the lazy array
        self._batches_generator = uproot.lazy(self.files, filter_name=self._variables_list,  cut=self.cut,
                                              step_size=int(self.specific_batch_size*1.5))

        # Work out how many batches there will actually generator. Make sure we don't miss too many events
        self._num_real_batches = nbatches
        test_arr = uproot.lazy(self.files, filter_name="TauJets." + self.dummy_var, step_size=10000,
                                       library="ak", cut=self.cut)
        self._num_real_batches = 0
        index = 0
        while True:
            end_point = index * self.specific_batch_size + self.specific_batch_size
            batch = test_arr[index * self.specific_batch_size: end_point]
            if len(batch) == 0:
                break
            index += 1
            self._num_real_batches += 1
        logger.log(f"Number of batches in {self.label} {self.data_type()} = {self._num_real_batches}", 'DEBUG')

        self._n_events_in_batch = 0

        logger.log(f"Found {len(files)} files for {data_type}", 'INFO')
        logger.log(f"Found these files: {files}", 'INFO')
        logger.log(f"Found {self._num_events} events for {data_type}", 'INFO')
        logger.log(f"DataLoader for {data_type} initialized", "INFO")

    def get_next_batch(self, idx=None):
        index = idx
        if idx is None or self._disable_indexing is True:
            index = self._current_index
        end_point = index*self.specific_batch_size + self.specific_batch_size
        batch = self._batches_generator[index*self.specific_batch_size: end_point]
        self._current_index += 1

        # If we run out of data reset the generator
        if len(batch) == 0:
            self._current_index = 0
            self._disable_indexing = True
            logger.log(f"{self.label} - {self._data_type} ran out of data - repeating data", 'DEBUG')
            return self.get_next_batch()

        return batch, np.ones(len(batch)) * self.class_label

    def get_batch_from_index(self, idx):
        end_point = idx*self.specific_batch_size + self.specific_batch_size
        if idx*self.specific_batch_size + self.specific_batch_size >= self.num_events():
            end_point = self.num_events() - 1
        batch = self._batches_generator[idx*self.specific_batch_size: end_point]
        return batch, np.ones(len(batch)) * self.class_label

    def pad_and_reshape_nested_arrays(self, batch, variable_type, max_items=20):
        variables = self._variables_dict[variable_type]
        ak_arrays = ak.concatenate([batch[var][:, :, None] for var in variables], axis=0)
        ak_arrays = ak.pad_none(ak_arrays, max_items, clip=True)
        np_arrays = ak.to_numpy(abs(ak_arrays))
        np_arrays = np_arrays.reshape(int(np_arrays.shape[0] / len(variables)), len(variables), np_arrays.shape[1])
        np_arrays = apply_logs(np_arrays, variables, log_list)
        return np_arrays

    def reshape_arrays(self, batch, variable_type):
        variables = self._variables_dict[variable_type]
        ak_arrays = ak.concatenate([batch[var][:] for var in variables], axis=0)
        np_arrays = ak.to_numpy(ak_arrays)
        np_arrays = np_arrays.reshape(int(np_arrays.shape[0] / len(variables)), len(variables))
        np_arrays = apply_logs(np_arrays, variables, log_list)
        return np_arrays

    def load_batch_from_data(self, idx=None):
        """
        Loads a batch of data of a specific data type. Pads ragged track and PFO arrays to make them rectilinear
        and reshapes arrays into correct shape for training. The clip option in ak.pad_none will truncate/extend each
        array so that they are all of a specific length- here we limit nested arrays to 20 items
        :param idx: The index of the batch to be processed
        :return: A list of arrays - [x1, x2, ... xn], labels, weight
        """

        batch, sig_bkg_labels_np_array = self.get_next_batch(idx)

        if batch is None or len(batch) == 0:
            logger.log(f"{self.label} - {self._data_type} ran out of data - repeating data", 'DEBUG')
            self._current_index = 0
            batch, sig_bkg_labels_np_array = self.get_next_batch()

        track_np_arrays = self.pad_and_reshape_nested_arrays(batch, "TauTracks", max_items=20)
        conv_track_np_arrays = self.pad_and_reshape_nested_arrays(batch, "ConvTrack", max_items=20)
        shot_pfo_np_arrays = self.pad_and_reshape_nested_arrays(batch, "ShotPFO", max_items=20)
        neutral_pfo_np_arrays = self.pad_and_reshape_nested_arrays(batch, "NeutralPFO", max_items=20)
        jet_np_arrays = self.reshape_arrays(batch, "TauJets")

        # Just do the 1-prong case for now
        labels_np_array = np.zeros((len(sig_bkg_labels_np_array), 4))
        if sig_bkg_labels_np_array[0] == 0:
            labels_np_array[:, 0] = 1
        else:
            truth_decay_mode_np_array = ak.to_numpy(batch[self._variables_dict["DecayMode"]])
            for i in range(0, len(truth_decay_mode_np_array, )):
                if truth_decay_mode_np_array[i][0] == 0:
                    labels_np_array[:, 1] = 1
                elif truth_decay_mode_np_array[i][0] == 1:
                    labels_np_array[:, 2] = 1
                else:
                    labels_np_array[:, 3] = 1

        # weight_np_array = ak.to_numpy(batch[self._variables_dict["Weight"]]).astype("float32")
        # Apply pT re-weighting
        weight_np_array = pt_reweight(ak.to_numpy(batch[self._variables_dict["TauJets.jet_pt"]]).astype("float32"))

        logger.log(f"Loaded batch {self._current_index} from {self._data_type}: {self.label}", "DEBUG")

        return (track_np_arrays, neutral_pfo_np_arrays, shot_pfo_np_arrays, conv_track_np_arrays, jet_np_arrays), \
                labels_np_array, weight_np_array

    def __next__(self):
        if self._current_index < self._num_real_batches:
            return self.load_batch_from_data()
        raise StopIteration

    def __len__(self):
        return self._num_real_batches

    def __iter__(self):
        return self

    def __call__(self):
        self._current_index = 0
        return self

    def __getitem__(self, idx):

        if idx < self._num_real_batches:
            return self.load_batch_from_data(idx=idx)

        elif self._current_index < self._num_real_batches:
            logger.log(f"Index out of bounds in __getitem__ - index {idx} is out of bounds for DataLoader of size "
                       f"{self._num_real_batches} - falling back to internal current_index = {self._current_index}",
                       'WARNING')
            return self.load_batch_from_data(idx=self._current_index)

        logger.log(f"Index out of bounds in __getitem__ - index {idx} is out of bounds for DataLoader of size "
                   f"{self._num_real_batches}", 'ERROR')
        raise IndexError

    def reset_index(self):
        self._current_index = 0
        self._disable_indexing = False
        self._batches_generator = None
        self._batches_generator = uproot.lazy(self.files, filter_name=self._variables_list, cut=self.cut,
                                              step_size=int(self.specific_batch_size * 1.5))

    def num_events(self):
        return self._num_events

    def data_type(self):
        return self._data_type

    def number_of_batches(self):
        return self._num_real_batches

    def get_batch_generator(self):
        return self._batches_generator


if __name__ == "__main__":
    pass

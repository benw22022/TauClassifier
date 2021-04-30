"""
DataLoader Class definition
________________________________________________________________________________________________________________________
A helper class to lazily load batches of training data for each type of data
"""

import math

import awkward as ak
import numpy as np
import uproot
from preprocessing import finite_log
from utils import logger
from variables import log_list


def apply_logs(np_arrays, variables, log_list):
    for i in range(0, len(variables)):
        if variables[i] in log_list:
            np_arrays[:, i] = np.log(np_arrays[:, i], out=np.zeros_like(np_arrays[:, i]), where=(np_arrays[:, i] > 1))
    return np_arrays



class DataLoader:

    def __init__(self, data_type, files, class_label, nbatches, variables_dict, dummy_var="truthProng", cuts=None,  batch_size=None,
                 label="Dataloader"):
        """
        Class constructor - fills in meta-data for the data type
        :param data_type: The type of data file being loaded e.g. Gammatautau, JZ1, ect...
        :param files: A list of files of the same data type to be loaded
        :param class_label: 1 for signal, 0 for background
        :param dummy_var: A variable to be loaded from the file to be loaded and iterated through to work out the number
        of events in the data files
        """
        self._data_type = data_type
        self.label = label
        self.files = files
        self.dummy_var = dummy_var
        self.cut = cuts
        self._nbatches = nbatches
        self.class_label = class_label
        self._variables_dict = variables_dict
        self._idx = 0


        dummy_array = uproot.concatenate(files, filter_name="TauJets."+dummy_var, cut=cuts)
        self._num_events = len(dummy_array["TauJets."+dummy_var])
        self._batch_len = 0

        # Set the DataLoader's batch size
        if batch_size is None:
            self.specific_batch_size = math.ceil(self._num_events / nbatches)
        else:
            self.specific_batch_size = batch_size


        self._variables_list = []
        for _, variable_list in variables_dict.items():
            self._variables_list += variable_list

        self._batches_generator = uproot.lazy(self.files, filter_name=self._variables_list, step_size=10000,
                                       library="ak", cut=self.cut)

        # Work out how many batches there will actually be in the iterator. Make sure we don't miss too many events
        self._num_real_batches = nbatches
        self._num_truncated_events = 0
        #for batch in uproot.iterate(self.files, filter_name="TauJets." + self.dummy_var, step_size=self.specific_batch_size,
        #                        library="ak", cut=self.cut):
        #    self._num_real_batches += 1
        #    if self._num_real_batches > self._nbatches:
        #        self._num_truncated_events += len(batch["TauJets." + self.dummy_var])


        #logger.log(f"Sample {self._data_type} has {self._num_real_batches} batches but is limited to only {self._nbatches} batches",'WARNING')
        #logger.log(f"Sample {self._data_type} will have {self._num_truncated_events} missing events. Consider tuning the DataLoader batch_size",'WARNING')

        self._n_events_in_batch = 0
        self._current_index = 0

        logger.log(f"Found {len(files)} files for {data_type}", 'INFO')
        logger.log(f"Found these files: {files}", 'INFO')
        logger.log(f"Found {self._num_events} events for {data_type}", 'INFO')
        logger.log(f"DataLoader for {data_type} initialized", "INFO")

    def set_batches(self,  variable_list):
        self._batches_generator = None
        #self._batches_generator = uproot.iterate(self.files, filter_name=variable_list, step_size=self.specific_batch_size,
        #                               library="ak", cut=self.cut, num_workers=12, num_fallback_workers=12, )
        self._batches_generator = uproot.lazy(self.files, filter_name=variable_list, step_size=10000,#step_size=self.specific_batch_size*2,
                                       library="ak", cut=self.cut)
        logger.log(f"Set batches for {self._data_type}")

    def get_next_batch(self):
        #batch = next(self._batches_generator)
        end_point = self._current_index*self.specific_batch_size + self.specific_batch_size
        batch = self._batches_generator[self._current_index*self.specific_batch_size: end_point]
        self._current_index += 1

        # If we run out of data reset the generator
        if batch is None:
            self._current_index = 0
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
        #np_arrays = finite_log(np_arrays)
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

    def load_batch_from_data(self, idx=0):
        """
        Loads a batch of data of a specific data type. Pads ragged track and cluster arrays to make them rectilinear
        and reshapes arrays into correct shape for training. The clip option in ak.pad_none will truncate/extend each
        array so that they are all of a specific length- here we require 20 tracks and 15 clusters.
        :param idx: The index of the batch to be processed
        :return: A list of arrays
            [Tracks, Clusters, Jets, Labels, Weight]
        """

        batch, sig_bkg_labels_np_array = self.get_next_batch()

        if batch is None or len(batch) == 0:
            logger.log(f"{self.label} - {self._data_type} ran out of data - repeating data", 'DEBUG')
            return None

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

        weight_np_array = ak.to_numpy(batch[self._variables_dict["Weight"]]).astype("float32")

        logger.log(f"Loaded batch {self._current_index} from {self._data_type}: {self.label}", "DEBUG")


        return (track_np_arrays, neutral_pfo_np_arrays, shot_pfo_np_arrays, conv_track_np_arrays, jet_np_arrays), \
               labels_np_array, weight_np_array

    def __next__(self):
        if self._current_index < self._num_real_batches:
            return self.load_batch_from_data()
        raise StopIteration

    def __len__(self):
        return self._batch_len

    def __iter__(self):
        return self

    def __call__(self):
        self._current_index = 0
        return self

    def __getitem__(self, idx):
        return self.load_batch_from_data(idx=idx)

    def reset_index(self):
        self._current_index = 0

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

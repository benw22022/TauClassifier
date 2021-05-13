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
import ray
import gc



@ray.remote
class DataLoader(object):

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
        self._current_batch = None

        # If we run out of data in this data loader
        self._disable_indexing = False

        # Parse variables
        self._variables_list = []
        for _, variable_list in variables_dict.items():
            self._variables_list += variable_list



        # # Work out how many batches there will actually generator. Make sure we don't miss too many events
        self._num_real_batches = nbatches
        test_arr = uproot.lazy(self.files, filter_name="TauJets." + self.dummy_var, step_size=10000, cut=self.cut)
        self._num_events = len(test_arr)
        self._num_real_batches = 0

        # Set the DataLoader's batch size
        if batch_size is None:
            self.specific_batch_size = math.ceil(self._num_events / nbatches)
        else:
            self.specific_batch_size = batch_size

        # Setup the lazy array
        self._batches_generator = uproot.iterate(self.files, filter_name=self._variables_list,  cut=self.cut,
                                                 step_size=self.specific_batch_size)

        index = 0
        while True:
            end_point = index * self.specific_batch_size + self.specific_batch_size
            batch = test_arr[index * self.specific_batch_size: end_point]
            if len(batch) == 0:
                break
            index += 1
            self._num_real_batches += 1
        # # logger.log(f"Number of batches in {self.label} {self.data_type()} = {self._num_real_batches}", 'DEBUG')
        #
        # logger.log(f"Found {len(files)} files for {data_type}", 'INFO')
        # logger.log(f"Found these files: {files}", 'INFO')
        # logger.log(f"Found {self._num_events} events for {data_type}", 'INFO')
        # logger.log(f"DataLoader for {data_type} initialized", "INFO")

    def get_next_batch(self, idx=None):
        index = idx
        if idx is None or self._disable_indexing is True:
            index = self._current_index
        end_point = index*self.specific_batch_size + self.specific_batch_size
        #batch = self._batches_generator[index*self.specific_batch_size: end_point]
        #batch = uproot.lazy(self.files, filter_name=self._variables_list, cut=self.cut)[index*self.specific_batch_size: end_point]
        try:
            batch = next(self._batches_generator)
        except StopIteration:
            self._batches_generator = uproot.iterate(self.files, filter_name=self._variables_list,  cut=self.cut,
                                                     library='ak', step_size=self.specific_batch_size)
        self._current_index += 1

        # If we run out of data reset the generator
        if len(batch) == 0:
            self._current_index = 0
            self._disable_indexing = True
            #logger.log(f"{self.label} - {self._data_type} ran out of data - repeating data", 'DEBUG')
            return self.get_next_batch()

        return batch, np.ones(len(batch)) * self.class_label

    def next_batch(self):
        batch = next(self._batches_generator)
        self._current_index += 1
        yield batch, np.ones(len(batch)) * self.class_label
    #
    # def get_batch_from_index(self, idx):
    #     end_point = idx*self.specific_batch_size + self.specific_batch_size
    #     if idx*self.specific_batch_size + self.specific_batch_size >= self.num_events():
    #         end_point = self.num_events() - 1
    #     batch = self._batches_generator[idx*self.specific_batch_size: end_point]
    #     return batch, np.ones(len(batch)) * self.class_label
    #
    def pad_and_reshape_nested_arrays(self, batch, variable_type, max_items=20):
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
        ak_arrays = ak.concatenate([batch[var][:, :, None] for var in variables], axis=0)
        ak_arrays = ak.pad_none(ak_arrays, max_items, clip=True)
        np_arrays = ak.to_numpy(abs(ak_arrays))
        np_arrays = np_arrays.reshape(int(np_arrays.shape[0] / len(variables)), len(variables), np_arrays.shape[1])
        #np_arrays = self.apply_scaling(np_arrays)
        return np_arrays

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
        ak_arrays = ak.concatenate([batch[var][:] for var in variables], axis=0)
        np_arrays = ak.to_numpy(abs(ak_arrays))
        np_arrays = np_arrays.reshape(int(np_arrays.shape[0] / len(variables)), len(variables))
        np_arrays = self.apply_scaling(np_arrays)
        return np_arrays
    #
    # def load_batch_from_data(self, idx=None):
    #     """
    #     Loads a batch of data of a specific data type. Pads ragged track and PFO arrays to make them rectilinear
    #     and reshapes arrays into correct shape for training. The clip option in ak.pad_none will truncate/extend each
    #     array so that they are all of a specific length- here we limit nested arrays to 20 items
    #     :param idx: The index of the batch to be processed
    #     :return: A list of arrays - [x1, x2, ... xn], labels, weight
    #     """
    #
    #     logger.log(f"{self.label} - Loading batch {idx} from {self._data_type}", "DEBUG")
    #
    #     batch, sig_bkg_labels_np_array = self.get_next_batch(idx)
    #
    #     if batch is None or len(batch) == 0:
    #         logger.log(f"{self.label} - {self._data_type} ran out of data - repeating data", 'DEBUG')
    #         self._current_index = 0
    #         batch, sig_bkg_labels_np_array = self.get_next_batch()
    #
    #     track_np_arrays = self.pad_and_reshape_nested_arrays(batch, "TauTracks", max_items=20)
    #     conv_track_np_arrays = self.pad_and_reshape_nested_arrays(batch, "ConvTrack", max_items=20)
    #     shot_pfo_np_arrays = self.pad_and_reshape_nested_arrays(batch, "ShotPFO", max_items=20)
    #     neutral_pfo_np_arrays = self.pad_and_reshape_nested_arrays(batch, "NeutralPFO", max_items=20)
    #     jet_np_arrays = self.reshape_arrays(batch, "TauJets")
    #
    #     # Just do the 1-prong case for now
    #     labels_np_array = np.zeros((len(sig_bkg_labels_np_array), 4))
    #     if sig_bkg_labels_np_array[0] == 0:
    #         labels_np_array[:, 0] = 1
    #     else:
    #         truth_decay_mode_np_array = ak.to_numpy(batch[self._variables_dict["DecayMode"]])
    #         for i in range(0, len(truth_decay_mode_np_array, )):
    #             if truth_decay_mode_np_array[i][0] == 0:
    #                 labels_np_array[i][1] = 1
    #             elif truth_decay_mode_np_array[i][0] == 1:
    #                 labels_np_array[i][2] = 1
    #             else:
    #                 labels_np_array[i][3] = 1
    #
    #     # Apply pT re-weighting
    #     weight_np_array = np.ones(len(labels_np_array))
    #     if self.class_label == 0:
    #         weight_np_array = pt_reweight(ak.to_numpy(batch[self._variables_dict["Weight"]]).astype("float32"))
    #
    #     logger.log(f"Loaded batch {self._current_index} from {self._data_type}: {self.label}", "DEBUG")
    #
    #     return (track_np_arrays, neutral_pfo_np_arrays, shot_pfo_np_arrays, conv_track_np_arrays, jet_np_arrays), \
    #             labels_np_array, weight_np_array
    #
    def apply_scaling(self, np_arrays):
        """
        Applies scaling to data to bring values into sensible ranges
        Checks:
            1. If the mean of the data is < 1 -> do nothing data is already fine
            2. Check if data is within 3 std of mean -> if not clip to mean + 3 std
            3. Divide data by mean value
            4. Take robust log of data -> if entry is < 1 do not take log but return 0 instead
        :param np_arrays:
        :return:
        """
        np_arrays = np.nan_to_num(np_arrays, posinf=0, neginf=0, copy=False)
        for i in range(0, np_arrays.shape[1]):
            mean = np.mean(abs(np_arrays[:, i]))
            std_dev = np.std(abs(np_arrays[:, i]))
            if mean > 1:
                np_arrays[:, i][abs(np_arrays[:, i]) > mean + 3 * std_dev] = mean + 5 * std_dev
                min_val = 0 #np.amin(np_arrays[:, i].flatten())
                max_val = mean + 5 * std_dev  #np.amax(np_arrays[:, i].flatten())
                np_arrays[:, i] = (np_arrays[:, i] - min_val) / (max_val - min_val)
                #np_arrays[:, i] = np.log10(np_arrays[:, i], out=np.zeros_like(np_arrays[:, i]), where=(np_arrays[:, i] > 1))
        return np_arrays
    #
    # # def __next__(self):
    # #     if self._current_index < self._num_real_batches:
    # #         return self.load_batch_from_data()
    # #     raise StopIteration
    #
    # def __len__(self):
    #     return self._num_real_batches
    #
    # # def __iter__(self):
    # #     return self
    # #
    # # def __call__(self):
    # #     self._current_index = 0
    # #     return self
    #
    # # def __getitem__(self, idx):
    # #
    # #     if idx < self._num_real_batches:
    # #         return self.load_batch_from_data(idx=idx)
    # #
    # #     elif self._current_index < self._num_real_batches:
    # #         logger.log(f"Index out of bounds in __getitem__ - index {idx} is out of bounds for DataLoader of size "
    # #                    f"{self._num_real_batches} - falling back to internal current_index = {self._current_index}",
    # #                    'WARNING')
    # #         return self.load_batch_from_data(idx=self._current_index)
    # #
    # #     logger.log(f"Index out of bounds in __getitem__ - index {idx} is out of bounds for DataLoader of size "
    # #                f"{self._num_real_batches}", 'ERROR')
    # #     raise IndexError
    #
    def reset_index(self):
        self._current_index = 0
        self._disable_indexing = False
        # self._batches_generator = uproot.lazy(self.files, filter_name=self._variables_list, cut=self.cut,
        #                                       step_size=int(self.specific_batch_size * 1.5))
        self._batches_generator = uproot.iterate(self.files, filter_name=self._variables_list, cut=self.cut,
                                                 library='ak', step_size=self.specific_batch_size)
    #
    def num_events(self):
        return self._num_events
    #
    def data_type(self):
        return self._data_type

    def number_of_batches(self):
        return self._num_real_batches
    #
    # def get_batch_generator(self):
    #     return self._batches_generator
    #
    def get(self):
        return self._current_batch

    def set_batch(self, idx):
        """
           Loads a batch of data of a specific data type and then stores it for later retrieval.
           Pads ragged track and PFO arrays to make them rectilinear
           and reshapes arrays into correct shape for training. The clip option in ak.pad_none will truncate/extend each
           array so that they are all of a specific length- here we limit nested arrays to 20 items
           :param idx: The index of the batch to be processed
           """

        #logger.log(f"{self.label} - Loading batch {idx} from {self._data_type}")

        batch, sig_bkg_labels_np_array = next(self.next_batch())

        if batch is None or len(batch) == 0:
            #logger.log(f"{self.label} - {self._data_type} ran out of data - repeating data", 'DEBUG')
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
                    labels_np_array[i][1] = 1
                elif truth_decay_mode_np_array[i][0] == 1:
                    labels_np_array[i][2] = 1
                else:
                    labels_np_array[i][3] = 1

        # Apply pT re-weighting
        weight_np_array = np.ones(len(labels_np_array))
        if self.class_label == 0:
            weight_np_array = pt_reweight(ak.to_numpy(batch[self._variables_dict["Weight"]]).astype("float32"))

        #logger.log(f"Loaded batch {self._current_index} from {self._data_type}: {self.label}", "DEBUG")
        #self._current_batch =

        self._current_batch = (track_np_arrays, neutral_pfo_np_arrays, shot_pfo_np_arrays, conv_track_np_arrays, jet_np_arrays),\
                                labels_np_array, weight_np_array

        # del track_np_arrays, neutral_pfo_np_arrays, shot_pfo_np_arrays, conv_track_np_arrays, jet_np_arrays, labels_np_array, weight_np_array
        # gc.collect()




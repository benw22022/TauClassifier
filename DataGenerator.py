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


class DataGenerator(keras.utils.Sequence):

    def __init__(self, file_dict, variables_dict, batch_size, cuts=None):
        """
        Class constructor - loads batches of data in a way that can be fed one by one to Keras - avoids having to load
        entire dataset into memory prior to training
        :param file_dict: A dictionary with keys labeling the data type (e.g. Gammatautau, JZ1, etc...) and values being
        a list of files corresponding to that data type
        :param variables_dict: A dictionary of input variables with keys labeling the variable type (Tracks, Clusters, etc...)
        and values being a list of branch names of the variables associated with that type
        :param batch_size: The number of events to train on per batch of data
        """
        print("Initializing DataGenerator")
        self._batch_size = batch_size
        self._file_dict = file_dict
        self.data_classes = []
        self.cuts = cuts

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
            self.data_classes.append(DataLoader(data_type, file_list, class_label, cut=self.cuts))

        # Get number of events in each dataset
        self.total_num_events = 0
        for data_class in self.data_classes:
            self.total_num_events += data_class.num_events

        # Lazily load batches of data for each dataset
        for data_class in self.data_classes:
            data_class.load_batches(self._variables_list, self.total_num_events, self._batch_size)

        # Work out the number of batches for training epoch (important)
        self._num_batches = len(self.data_classes[0].batches)

        print("DataGenerator initialized")

    def pad_and_reshape_nested_arrays(self, batch, variable_type, max_items=20):

        variables = [v.replace(f"{variable_type}.", "") for v in self._variables_dict[variable_type]]
        ak_arrays = ak.concatenate([batch[variable_type][var][:, :, None] for var in variables], axis=0)
        ak_arrays = ak.pad_none(ak_arrays, max_items, clip=True)
        np_arrays = ak.to_numpy(ak_arrays)
        np_arrays = np_arrays.reshape(int(np_arrays.shape[0] / len(variables)), len(variables), np_arrays.shape[1])
        return np_arrays

    def reshape_arrays(self, batch, variable_type):
        variables = self._variables_dict[variable_type]
        ak_arrays = ak.concatenate([batch[var][:] for var in variables], axis=0)
        np_arrays = ak.to_numpy(ak_arrays)
        np_arrays = np_arrays.reshape(int(np_arrays.shape[0] / len(variables)), len(variables))
        return np_arrays

    def _load_batch_from_data(self, data_class, idx):
        """
        Loads a batch of data of a specific data type. Pads ragged track and cluster arrays to make them rectilinear
        and reshapes arrays into correct shape for training. The clip option in ak.pad_none will truncate/extend each
        array so that they are all of a specific length- here we require 20 tracks and 15 clusters.
        :param data_class: Data type to be loaded - e.g. Gammatautau, JZ1, etc...
        :param idx: The index of the batch to be processed
        :return: A list of arrays
            [Tracks, Clusters, Jets, Labels, Weight]
        """
        batch = data_class.batches[idx]

        print(batch["TauTracks"])


        track_np_arrays = self.pad_and_reshape_nested_arrays(batch, "TauTracks", max_items=20)
        conv_track_np_arrays = self.pad_and_reshape_nested_arrays(batch, "ConvTrack", max_items=20)
        shot_pfo_np_arrays = self.pad_and_reshape_nested_arrays(batch, "ShotPFO", max_items=20)
        neutral_pfo_np_arrays = self.pad_and_reshape_nested_arrays(batch, "NeutralPFO", max_items=20)
        jet_np_arrays = self.reshape_arrays(batch, "TauJets")

        # Just do the 1-prong case for now
        sig_bkg_labels_np_array = data_class.batch_labels(idx)
        labels_np_array = np.zeros((len(sig_bkg_labels_np_array), 4), dtype="object")
        if sig_bkg_labels_np_array[0] == 0:
            labels_np_array[:] = np.array([1, 0, 0, 0])
        else:
            truth_decay_mode_np_array = ak.to_numpy(batch[self._variables_dict["DecayMode"]])
            for i in range(0, len(truth_decay_mode_np_array,)):
                if truth_decay_mode_np_array[i] == 0:
                    labels_np_array[i] = np.array([0, 1, 0, 0])
                elif truth_decay_mode_np_array[i] == 1:
                    labels_np_array[i] = np.array([0, 0, 1, 0])
                else:
                    labels_np_array[i] = np.array([0, 0, 0, 1])

        weight_np_array = ak.to_numpy(batch[self._variables_dict["Weight"]])

        return track_np_arrays, conv_track_np_arrays, shot_pfo_np_arrays, neutral_pfo_np_arrays, jet_np_arrays, \
               labels_np_array, weight_np_array

    def load_batch(self, idx):
        """
        Loads a batch of data from each data type and concatenates them into single arrays for training
        :param idx: The index of the batch of data to retrieve
        :return: A list of arrays to be passed to model.fit()
            [[Tracks, Clusters, Jets], labels, weights]
        """
        track_array = np.array([])
        #cluster_array = np.array([])
        conv_track_array = np.array([])
        shot_pfo_array = np.array([])
        neutral_pfo_array = np.array([])
        jet_array = np.array([])
        label_array = np.array([])
        weight_array = np.array([])

        for i in range(0, len(self.data_classes)):
            if i == 0:
                track_array, conv_track_array, shot_pfo_array, neutral_pfo_array, jet_array, \
                label_array, weight_array = self._load_batch_from_data(self.data_classes[i], idx)
            else:
                tmp_track_array, tmp_conv_track_array, tmp_shot_pfo_array, tmp_neutral_pfo_array, tmp_jet_array, \
                tmp_label_array, tmp_weight_array = self._load_batch_from_data(self.data_classes[i], idx)
                track_array = np.concatenate((tmp_track_array, track_array))
                conv_track_array = np.concatenate((tmp_conv_track_array, conv_track_array))
                shot_pfo_array = np.concatenate((tmp_shot_pfo_array, shot_pfo_array))
                neutral_pfo_array = np.concatenate((tmp_neutral_pfo_array, neutral_pfo_array))
                #cluster_array = np.concatenate((tmp_cluster_array, cluster_array))
                jet_array = np.concatenate((tmp_jet_array, jet_array))
                label_array = np.concatenate((tmp_label_array, label_array))
                weight_array = np.concatenate((tmp_weight_array, weight_array))

        return [track_array, conv_track_array, shot_pfo_array, neutral_pfo_array, jet_array], label_array, weight_array

    def get_batch_shapes(self, idx=0):
        """
        Loads a batch at a specific index and returns the shapes of the returned arrays
        Used for debugging and initializing network input layers
        :param idx: An index (default is zero)
        :return: A list of all the array shapes.
            tracks.shape, clusters.shape, jets.shape, labels.shape, weights.shape
        """
        batch = self.load_batch(idx)
        shapes = []

        for item in batch[0]:
            shapes.append(item.shape)
        shapes.append(batch[1].shape)
        shapes.append(batch[2].shape)

        return shapes

    def __len__(self):
        """
        This returns the number of batches that the data was split up into
        :return: The number of batches in an epoch
        """
        return self._num_batches

    def __getitem__(self, idx):
        """
        The function that Keras will call in order to get a new batch of data to train on. Same as self.load_batch().
        :param idx: An index - set by Keras
        :return: A full batch of data
        """
        return self.load_batch(idx)

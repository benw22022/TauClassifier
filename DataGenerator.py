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

    def __init__(self, file_dict, variables_dict, batch_size):
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

        # Organise variable list
        self._variables_dict = variables_dict
        self._variables_list = []
        for _, variable_list in variables_dict.items():
            self._variables_list += variable_list

        # Load each data type separately
        for data_type, file_list in file_dict.items():
            class_label = 0
            if data_type == "Gammatautau":
                class_label = 1
            self.data_classes.append(DataLoader(data_type, file_list, class_label))

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

        track_vars = [v.replace("TauTracks.", "") for v in self._variables_dict["Tracks"]]
        track_ak_arrays = ak.concatenate([batch["TauTracks"][var][:, :, None] for var in track_vars], axis=0)
        track_ak_arrays = ak.pad_none(track_ak_arrays, 20, clip=True)
        track_np_arrays = ak.to_numpy(track_ak_arrays)
        track_np_arrays = track_np_arrays.reshape(int(track_np_arrays.shape[0] / len(track_vars)), len(track_vars),
                                                  track_np_arrays.shape[1])

        cluster_vars = [v.replace("TauClusters.", "") for v in self._variables_dict["Clusters"]]
        cluster_ak_arrays = ak.concatenate([batch["TauClusters"][var][:, :, None] for var in cluster_vars], axis=0)
        cluster_ak_arrays = ak.pad_none(cluster_ak_arrays, 15, clip=True)
        cluster_np_arrays = ak.to_numpy(cluster_ak_arrays)
        cluster_np_arrays = cluster_np_arrays.reshape(int(cluster_np_arrays.shape[0] / len(cluster_vars)), len(cluster_vars),
                                                      cluster_np_arrays.shape[1])

        jet_vars = self._variables_dict["Jets"]
        jet_ak_arrays = ak.concatenate([batch[var][:] for var in jet_vars], axis=0)
        jet_np_arrays = ak.to_numpy(jet_ak_arrays)
        jet_np_arrays = jet_np_arrays.reshape(int(jet_np_arrays.shape[0] / len(jet_vars)), len(jet_vars))

        labels_np_array = data_class.batch_labels(idx)

        weight_np_array = ak.to_numpy(batch["TauJets.mcEventWeight"])

        return track_np_arrays, cluster_np_arrays, jet_np_arrays, labels_np_array, weight_np_array

    def load_batch(self, idx):
        """
        Loads a batch of data from each data type and concatenates them into single arrays for training
        :param idx: The index of the batch of data to retrieve
        :return: A list of arrays to be passed to model.fit()
            [[Tracks, Clusters, Jets], labels, weights]
        """
        track_array = np.array([])
        cluster_array = np.array([])
        jet_array = np.array([])
        label_array = np.array([])
        weight_array = np.array([])

        for i in range(0, len(self.data_classes)):
            if i == 0:
                track_array, cluster_array, jet_array, label_array, weight_array = self._load_batch_from_data(
                    self.data_classes[i], idx)
            else:
                tmp_track_array, tmp_cluster_array, tmp_jet_array, tmp_label_array, tmp_weight_array = self._load_batch_from_data(
                    self.data_classes[i], idx)
                track_array = np.concatenate((tmp_track_array, track_array))
                cluster_array = np.concatenate((tmp_cluster_array, cluster_array))
                jet_array = np.concatenate((tmp_jet_array, jet_array))
                label_array = np.concatenate((tmp_label_array, label_array))
                weight_array = np.concatenate((tmp_weight_array, weight_array))

        return [track_array, cluster_array, jet_array], label_array, weight_array

    def get_batch_shapes(self, idx=0):
        """
        Loads a batch at a specific index and returns the shapes of the returned arrays
        Used for debugging and initializing network input layers
        :param idx: An index (default is zero)
        :return: A list of all the array shapes.
            tracks.shape, clusters.shape, jets.shape, labels.shape, weights.shape
        """
        batch = self.load_batch(idx)
        return batch[0][0].shape, batch[0][1].shape, batch[0][2].shape, batch[1].shape, batch[2].shape

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







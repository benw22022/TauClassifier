"""
DataLoader Class definition
________________________________________________________________________________________________________________________
A helper class and ray actor to generate batches of data from a set of root files. This object applies all the required
transformations to the data and computes the class labels.
"""

import math
import os.path
import awkward as ak
import numpy as np
import uproot
import ray
import gc
import numba as nb
from scripts.utils import logger, profile_memory
from config.config import models_dict


@nb.njit()
def labeler(truth_decay_mode_np_array, labels_np_array, prong=None):
    """
    Function to compute decay mode labels for Gammatautau. Due to large for loop, the function is jitted for speed.
    This function would ideally be a member of DataLoader but isn't since jitting member functions is hard
    :param truth_decay_mode_np_array: The Truth Decay Mode - an enum corresponding to the decay mode
        - 1p0n == 0
        - 1p1n == 1
        - 1pXn == 2
        - 3p0n == 3
        - 3pXn == 4
    :param labels_np_array: The array of labels - already initialized ready to be modified
    Convention:
    [1, 0, 0, 0, 0, 0] == Background Jets
    [0, 1, 0, 0, 0, 0] == 1p0n
    [0, 0, 1, 0, 0, 0] == 1p1n
    [0, 0, 0, 1, 0, 0] == 1pXn
    [0, 0, 0, 0, 1, 0] == 3p0n
    [0, 0, 0, 0, 0, 1] == 3pXn
    :return: An array of labels
    """
    for i in range(0, len(truth_decay_mode_np_array, )):
        elem = truth_decay_mode_np_array[i]
        if prong is None:
            labels_np_array[i][elem + 1] = 1
        elif prong == 3:
            labels_np_array[i][elem - 4 + 1] = 1
    return labels_np_array


@ray.remote
class DataLoader:

    def __init__(self, data_type, files, class_label, nbatches, variable_handler, dummy_var="truthProng", cuts=None,
                 batch_size=None, prong=None, reweighter=None, label="Dataloader", no_gpu=False):
        """
        Class constructor for the DataLoader object. Object is decorated with @ray.remote for easy multiprocessing
        To initialize the class (which is a ray actor) do: dl = Dataloader.remote(*args, **kwargs)
        To call a class method do: dl.<method>.remote(*args, **kwargs) - this returns a ray futures object
        To gather results of a class method do: ray.get(dl.<method>.remote(*args, **kwargs))
        :param data_type: A string labelling the data type e.g. Gammatautau, JZ1 etc..
        :param files: A list of file paths to NTuples to read from
        :param class_label: Either 0 for jets or 1 for taus
        :param nbatches: Number of batches to roughly split the data into - true number of batches will vary due to the
        way that uproot works - it cannot make batches split across two files
        :param variables_dict: A dictionary whose keys correspond to variable types e.g. TauTracks, NeutralPFO etc...
        and whose values are a list of branches belonging to that key type
        :param dummy_var: A branch that can be easily loaded for computing the number of events in a sample - don't use
        a nested variable (e.g. TauTracks.pt) as this will be slow and may cause an OOM error
        :param cuts: A string detailing the cuts to be applied to the data, passable by uproot
        e.g.(TauJets.ptJetSeed > 15000.0) & (TauJets.ptJetSeed < 10000000.0)
        :param batch_size: The number of events to load per batch (overrides nbatches opt.)
        :param num_classes: Number of prongs
        :param reweighter: An instance of a reweighting class 
        :param label:
        :param no_gpu:
        """
        # Disables GPU - useful if you want to instantiate multiple tensorflow model instances
        if no_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self._data_type = data_type
        self.label = label
        self.files = files
        self.dummy_var = dummy_var
        self.cut = cuts
        self._nbatches = nbatches
        self.class_label = class_label
        self._variable_handler = variable_handler
        self._current_index = 0
        self._reweighter = reweighter

        # Number of classes
        self._prong = prong
        self._nclasses = 6  # [1p0n, 1p1n, 1pxn, 3p0n, 3p1n, jets]
        if prong == 1:
            self._nclasses = 4  # [1p0n, 1p1n, 1pxn, jets]
        elif prong == 3:
            self._nclasses = 3  # [3p0n, 3pxn, jets]

        # Work out how many events there in the sample by loading up a small array
        test_arr = uproot.concatenate(self.files, filter_name="TauJets." + self.dummy_var, cut=self.cut, library='np')
        self._num_events = len(test_arr["TauJets." + self.dummy_var])

        # Set the DataLoader's batch size
        if batch_size is None:
            self.specific_batch_size = math.ceil(self._num_events / nbatches)
        else:
            self.specific_batch_size = batch_size

        # Setup the iterator
        self._batches_generator = uproot.iterate(self.files, filter_name=self._variable_handler.list(), cut=self.cut,
                                                 step_size=self.specific_batch_size)

        # Work out the number of batches there are in the generator
        self._num_real_batches = 0
        for _ in uproot.iterate(self.files, filter_name="TauJets." + self.dummy_var, cut=self.cut,
                                step_size=self.specific_batch_size):
            self._num_real_batches += 1

        logger.log(f"Found {len(files)} with files {self._num_events} events for {data_type}", 'INFO')
        logger.log(f"Found these files: {files}", 'DEBUG')
        logger.log(f"Number of batches in {self.label} {self.data_type()} = {self._num_real_batches}", 'DEBUG')
        logger.log(f"DataLoader for {data_type} initialized", "DEBUG")

    def next_batch(self):
        """
        Gets the next batch of data from iterator. If end of the iterator is reached
        then restart it
        :return: batch - a dict of arrays yielded by uproot.iterate()
        """
        try:
            batch = next(self._batches_generator)
        except StopIteration:
            self._batches_generator = uproot.iterate(self.files, filter_name=self._variable_handler.list(), cut=self.cut,
                                                     step_size=self.specific_batch_size)
            return self.next_batch()
        self._current_index += 1
        return batch    

    def pad_and_reshape_nested_arrays(self, batch, variable_type, max_items=10, shuffle_var=None):
        """
        Function that acts on nested data to read relevant variables, pad, reshape and convert data from uproot into
        rectilinear numpy arrays
        :param batch (dict): A dict of awkward arrays from uproot
        :param variable_type (str): Variable type to be selected e.g. Tracks, Neutral PFO, Jets etc...
        :param max_items (int): Maximum number of tracks/PFOs etc... to be associated to event
        :param shuffle_var (str): When permutation ranking Variable to shuffle 
        :return np_arrays: a rectilinear numpy array of shape:
                (num events in batch, number of variables belonging to variable type, max_items)
        """
        variables = self._variable_handler.get(variable_type)
        np_arrays = np.zeros((ak.num(batch[variables[0].name], axis=0), len(variables), max_items))
        dummy_val = -1
        for i, variable in enumerate(variables):
            ak_arr = batch[variable.name]
            ak_arr = ak.pad_none(ak_arr, max_items, clip=True, axis=1)
            arr = ak.to_numpy(abs(ak_arr)).filled(dummy_val)
            arr = variable.standardise(arr, dummy_val=dummy_val)
            if variable.name == shuffle_var:
                np.random.shuffle(arr)            
            np_arrays[:, i] = arr
        np_arrays = np.nan_to_num(np_arrays, posinf=0, neginf=0, copy=False).astype("float32")
        return np_arrays

    def reshape_arrays(self, batch, variable_type, shuffle_var=None):
        """
        Function that acts on flat data to read relevant variables, reshape and convert data from uproot into
        rectilinear numpy arrays
        :param batch: A dict of awkward arrays from uproot
        :param variable_type: Variable type to be selected e.g. Tracks, Neutral PFO, Jets etc...
        :return: a rectilinear numpy array of shape:
                (num events in batch, number of variables belonging to variable type)
        """
        variables = self._variable_handler.get(variable_type)
        np_arrays = np.zeros((ak.num(batch[variables[0].name], axis=0), len(variables)))

        for i, variable in enumerate(variables):
            ak_arr = batch[variable.name]
            arr = ak.to_numpy(abs(ak_arr))
            arr = variable.standardise(arr)
            if variable.name == shuffle_var:
                np.random.shuffle(arr)
            np_arrays[:, i] = arr
        np_arrays = np.nan_to_num(np_arrays, posinf=0, neginf=0, copy=False).astype("float32")
        return np_arrays

    def get_batch(self, shuffle_var=None):
        """
        Loads a batch of data of a specific data type and then stores it for later retrieval.
        Pads ragged track and PFO arrays to make them rectilinear
        and reshapes arrays into correct shape for training. The clip option in ak.pad_none will truncate/extend each
        array so that they are all of a specific length
        :param shuffle_var (optional, default=None): A variable to shuffle (for permutation ranking)
        """
        batch = self.next_batch()

        track_np_arrays = self.pad_and_reshape_nested_arrays(batch, "TauTracks", max_items=5, shuffle_var=shuffle_var)
        conv_track_np_arrays = self.pad_and_reshape_nested_arrays(batch, "ConvTrack", max_items=5, shuffle_var=shuffle_var)
        shot_pfo_np_arrays = self.pad_and_reshape_nested_arrays(batch, "ShotPFO", max_items=5, shuffle_var=shuffle_var)
        neutral_pfo_np_arrays = self.pad_and_reshape_nested_arrays(batch, "NeutralPFO", max_items=5, shuffle_var=shuffle_var)
        jet_np_arrays = self.reshape_arrays(batch, "TauJets", shuffle_var=shuffle_var)

        # Compute labels
        labels_np_array = np.zeros((len(batch), self._nclasses))
        if self.class_label == 0:
            labels_np_array[:, 0] = 1
        else:
            truth_decay_mode_np_array = ak.to_numpy(batch["TauJets.truthDecayMode"]).astype(np.int64)
            labels_np_array = labeler(truth_decay_mode_np_array, labels_np_array, prong=self._prong)

        # Apply pT re-weighting
        weight_np_array = np.ones(len(labels_np_array))
        if self.class_label == 0:
            weight_np_array = self._reweighter.reweight(ak.to_numpy(batch["TauJets.ptJetSeed"]).astype("float32"))

        result = ((track_np_arrays, neutral_pfo_np_arrays, shot_pfo_np_arrays, conv_track_np_arrays, jet_np_arrays),
                  labels_np_array, weight_np_array)
        try:
            return result
        finally:
            del result

    def reset_dataloader(self):
        """
        Resets the DataLoader by restarting its index and iterator
        :return:
        """
        self._current_index = 0
        self._batches_generator = uproot.iterate(self.files, filter_name=self._variable_handler.list(), cut=self.cut,
                                                 library='ak', step_size=self.specific_batch_size)        
        gc.collect()

    def _set_generator_to_single_file(self, file, cut=None):
        """
        Function to set the uproot iterator. Definitely a bit hacky!
        The idea is that we can set the iterator to iterate over just a single file for when we want to 
        update the scores in the NTuples.
        :param file (str): file path to an NTuple
        :param cut (str: optional - default=None): A string defining cuts
        """
        self._batches_generator = uproot.iterate(file, filter_name=self._variable_handler.list(), cut=cut,
                                                 library='ak', step_size=self.specific_batch_size)

    def num_events(self):
        return self._num_events

    def data_type(self):
        return self._data_type

    def number_of_batches(self):
        return self._num_real_batches

    def predict(self, model, model_config, model_weights, file=None, save_predictions=False):
        """
        Function to generate arrays of y_pred, y_true and weights given a network weight file
        :param model: A string of corresponding to a key in config.config.models_dict specifiying desired model
        :param model_config: model config dictionary to use
        :param model_weights: Path to model weights file to load
        :param save_predictions (optional, default=False): write y_pred, y_true and weights to file
        :param file (optional, default=None): A file path to a single NTuple
        """

        # Set generator to use just a single file. Used when writing predictions to NTuples
        if file is not None:
            self._set_generator_to_single_file(files=file)

        # Model needs to be initialized on each actor separately - cannot share model between multiple processes
        model = models_dict[model](model_config)
        model.load_weights(model_weights)

        # Allocate arrays for y_pred, y_true and weights
        y_pred = np.ones((self.num_events(), self._nclasses)) * -999  # multiply by -999 so mistakes are obvious
        y_true = np.ones((self.num_events(), self._nclasses)) * -999
        weights = np.ones((self.num_events())) * -999
        nevents = 0

        # Iterate through the DataLoader
        position = 0
        for i in range(0, self._num_real_batches):
            batch, truth_labels, batch_weights = self.get_batch()
            nevents += len(truth_labels)
        
            # Fill arrays
            y_pred[position: position + len(batch[1])] = model.predict(batch)
            y_pred[position: position + len(batch[1])] = truth_labels
            weights[position: position + len(batch[1])] = batch_weights

            # Move to the next position
            position += len(batch[1])
            logger.log(f"{self._data_type} -- predicted batch {i}/{self._num_real_batches}")

        # Truncate arrays to get rid of garbage
        y_pred = y_pred[ :nevents]
        y_true = y_true[ :nevents]
        weights = weights[ :nevents]

        # Save the predictions, truth and weight to file
        if save_predictions:
            save_file = os.path.basename(str(self.files))
            np.savez(f"network_predictions/predictions/{save_file}_predictions.npz", y_pred)
            np.savez(f"network_predictions/truth/{save_file}_truth.npz", y_true)
            np.savez(f"network_predictions/weights/{save_file}_weights.npz", weights)
            logger.log(f"Saved network predictions for {self._data_type}")

        # Reset DataLoader
        self.reset_dataloader()

        return y_pred, y_true, weights

    def update_ntuples(self, model, model_config, model_weights):
        """
        Update the NTuples belonging to this DataLoader
        """

        for file in self.files:
            y_pred, _, weights = self.predict(self, model, model_config, model_weights, file=file)
            with uproot.update(self.file[0]) as ntuple:
                logger.log(f"Writing predictions for {file} to NTuple")
                ntuple['tree'] = ({"TauClassifier_jetScore": y_pred[:, 0],
                                        "TauClassifier_1p0nScore": y_pred[:, 1],
                                        "TauClassifier_1p1nScore": y_pred[:, 2],
                                        "TauClassifier_1pxnScore": y_pred[:, 3],
                                        "TauClassifier_3p0nScore": y_pred[:, 4],
                                        "TauClassifier_3pxnScore": y_pred[:, 5],
                                        "TauClassifier_weights": weights})
    
    def get_memory_profile(self):
        return profile_memory(self)
            
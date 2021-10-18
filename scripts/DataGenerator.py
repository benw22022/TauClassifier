"""
DataGenerator Class Definition
________________________________________________________________________________________________________________________
Class that is used to feed Keras batches of data for training/testing/validation so that we don't have to load all the
data into memory at once
TODO: Make this generalisable to different problems
"""

import os
import numpy as np
import keras
import tensorflow as tf
import datetime
import time
import gc
import ray  
from scripts.DataLoader import DataLoader, apply_scaling
from plotting.plotting_functions import plot_confusion_matrix
from scripts.utils import logger, find_anomalous_entries
from config.config import models_dict


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, file_handler_list, variables_dict, nbatches=1000, cuts=None, label="DataGenerator", reweighter=None,
                prong=None, no_gpu=False, _benchmark=False):
        """
        Class constructor for DataGenerator. Inherits from keras.utils.Sequence. When passed to model.fit(...) loads a
        batch of data from file for the network to train on. This avoids having to load large amounts of data into
        memory. To speed up I/O data can be read using multiple threads - this is achieved using the ray library (see:
        https://docs.ray.io/en/master/index.html). Each file stream is read in parallel by a DataLoader object which is
        instantiated on a new thread as a ray actor. The batches loaded by the DataLoaders are then gathered and
        merged together by this class.

        :param file_handler_list: A list of FileHandler Objects (a utility class defined in utils.py), each FileHandler
        in the list will create a new ray actor to handle their files
        :param variables_dict: A dictionary whose keys correspond to variable types e.g. TauTracks, NeutralPFO etc...
        and whose values are a list of branches belonging to that key type
        :param nbatches: Number of batches to roughly split the data into - true number of batches will vary due to the
        way that uproot works - it cannot make batches split across two files
        :param cuts: A dictionary whose keys match the FileHandler labels and whose values are strings that can be
        interpreted by uproot as a cut e.g. "(pt1 > 50) & ((E1>100) | (E1<90))"
        :param label: A label to give this object (helpful if you have multiple DataGenerators running at once)
        :param reweighter: An instance of a reweighting class 
        :param prong: Number of prongs - either 1-prong with 4 classes, 3-prong with 3 classes or 1+3-prong for 6 classes
        :param no_gpu: If True will make TensorFlow use CPU rather than GPU - useful when creating multiple models  
        :param _benchmark: If set to True will return additional information when load_batch() is called. This will
        cause model.fit() to break and is only used for testing purposes
        """

        logger.log(f"Initializing DataGenerator: {label}", 'INFO')
        
        # Disables GPU - useful if you want to instantiate multiple tensorflow model instances
        if no_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self._file_handlers = file_handler_list
        self.label = label
        self.data_loaders = []
        self.cuts = cuts
        self._current_index = 0
        self._epochs = 0
        self.__benchmark = _benchmark
        self.model = None
        self.prong = prong

        # Organise a list of all variables
        self._variables_dict = variables_dict
        self._variables_list = []
        for _, variable_list in variables_dict.items():
            self._variables_list += variable_list

        # Initialize ray actors from FileHandlers, variables_dict and cuts
        for file_handler in self._file_handlers:
            if cuts is not None and file_handler.label in cuts:
                logger.log(f"Cuts applied to {file_handler.label}: {self.cuts[file_handler.label]}")

                dl_label = file_handler.label + "_" + self.label
                file_list = file_handler.file_list
                class_label = file_handler.class_label
                dl = DataLoader.remote(file_handler.label, file_list, class_label, nbatches, variables_dict, cuts=self.cuts[file_handler.label],
                                                    label=label, prong=prong, reweighter=reweighter)
                self.data_loaders.append(dl)

            else:
                dl_label = file_handler.label + "_" + self.label
                file_list = file_handler.file_list
                class_label = file_handler.class_label
                dl = DataLoader.remote(file_handler.label, file_list, class_label, nbatches, variables_dict, prong=prong, label=dl_label, reweighter=reweighter)
                self.data_loaders.append(dl)

        # Get number of events in each dataset
        self._total_num_events = []
        num_batches_list = []
        for data_loader in self.data_loaders:
            self._total_num_events.append(ray.get(data_loader.num_events.remote()))
            num_batches_list.append(ray.get(data_loader.number_of_batches.remote()))
        self._total_num_events = sum(self._total_num_events)
        logger.log(f"{self.label} - Found {self._total_num_events} events total", "INFO")

        # Work out how many batches to split the data into
        self._num_batches = min(num_batches_list)

        # Work out number of classes
        self._nclasses = 6
        if prong == 1:
            self._nclasses = 4
        if prong == 3:
            self.nclasses = 3

    def load_batch(self, shuffle_var=""):
        """
        Loads a batch of data from each DataLoader and concatenates them into single arrays for training
        :return: A list of arrays to be passed to model.fit()
        """

        batch_load_time = time.time()

        batch = ray.get([dl.get_batch.remote(shuffle_var=shuffle_var) for dl in self.data_loaders])

        track_array = np.concatenate([result[0][0] for result in batch])
        neutral_pfo_array = np.concatenate([result[0][1] for result in batch])
        shot_pfo_array = np.concatenate([result[0][2] for result in batch])
        conv_track_array = np.concatenate([result[0][3] for result in batch])
        jet_array = np.concatenate([result[0][4] for result in batch])
        label_array = np.concatenate([result[1] for result in batch])
        weight_array = np.concatenate([result[2] for result in batch])

        load_time = str(datetime.timedelta(seconds=time.time()-batch_load_time))
        logger.log(f"{self.label}: Processed batch {self._current_index}/{self.__len__()} - {len(label_array)} events"
                   f" in {load_time}", "DEBUG")

        if self.__benchmark:
            return ((track_array, neutral_pfo_array, shot_pfo_array, conv_track_array, jet_array), label_array, weight_array),\
                    label_array, self._current_index, load_time

        try:
            return (track_array, neutral_pfo_array, shot_pfo_array, conv_track_array, jet_array), label_array, weight_array
        finally:
            # A (probably vain) attempt to free memory (The finally block allows you to run something after function returns)
            del batch
            del track_array, neutral_pfo_array, shot_pfo_array, conv_track_array, jet_array, label_array, weight_array
            gc.collect()

    def load_model(self, model, model_config, model_weights):
        """
        Function to set the model to be used by the DataGenerator for predictions
        Seems to me to be more efficient to store this as a class member rather than reinitialising the model
        everytime predict() is called
        Note: You should set no_gpu=True if you load models on multiple DataGenerator instances (TensorFlow is likely to complain)
        """
        self.model = models_dict[model](model_config)
        self.model.load_weights(model_weights)

    def predict(self, model=None, model_config=None, model_weights=None, save_predictions=False, shuffle_var="", make_confusion_matrix=False):
        """
        Function to generate arrays of y_pred, y_true and weights given a network weight file
        :param model: A string of corresponding to a key in config.config.models_dict specifiying desired model
        :param model_config: model config dictionary to use
        :param model_weights: Path to model weights file to load
        :param save_predictions (optional, default=False): write y_pred, y_true and weights to file
        :param shuffle_var (optional, default=""): This variable will be shuffled when generating each batch - this can be used
        for permutation variable ranking. Shuffling a more important variable will lead to a larger change in loss/accuracy
        """

        # Initialise model if it hasn't been already
        if self.model is None:
            if model is None or model_config is None or model_weights is None:
                logger.log(f"No model has been initialised on {self.label}! You must pass a model, model_config and model_weights to predict", "ERROR")
                logger.log("Alternativly you may pass those arguements to load_model() to set the model seperately", "ERROR")
                raise ValueError 
            self.load_model(model, model_config, model_weights)

        # Initialise loss and accuracy classes
        cce_loss = tf.keras.losses.CategoricalCrossentropy()
        acc_metric = tf.keras.metrics.Accuracy()

        # Allocate arrays for y_pred, y_true and weights
        y_pred = np.ones((self._total_num_events, self._nclasses)) * -999  # multiply by -999 so mistakes are obvious
        y_true = np.ones((self._total_num_events, self._nclasses)) * -999
        weights = np.ones((self._total_num_events)) * -999
        losses = []
        accs = []

        # Iterate through the DataLoader
        position = 0
        for i in range(0, self.__len__()):
            batch, truth_labels, batch_weights = self.load_batch(shuffle_var=shuffle_var)
            predicted_labels = self.model.predict(batch)
            loss = cce_loss(truth_labels, predicted_labels).numpy()
            acc_metric.update_state(truth_labels, predicted_labels, batch_weights) 
            acc = acc_metric.result().numpy()
            self._current_index += 1
            try:
                # Fill arrays
                y_pred[position: position + len(batch[1])] = predicted_labels
                y_pred[position: position + len(batch[1])] = truth_labels
                weights[position: position + len(batch[1])] = batch_weights
                losses.append(loss)
                accs.append(acc)
            except ValueError:
                # If we overstep the end of the array - fill in the last few entries
                y_pred[position:] = predicted_labels
                y_pred[position:] = truth_labels
                weights[position:] = batch_weights
                losses.append(loss)
                accs.append(acc)

            # Move to the next position
            position += len(batch[1])

        # Save the predictions, truth and weights to file
        if save_predictions:
            if saveas is None:
                save_file = os.path.basename(self.files[0])
                np.savez(f"network_predictions/predictions/{save_file}_predictions.npz", y_pred)
                np.savez(f"network_predictions/truth/{save_file}_truth.npz", y_true)
                np.savez(f"network_predictions/weights/{save_file}_weights.npz", weights)
                logger.log(f"Saved network predictions for {self._data_type}")
        
        # Plot confusion matrix if requested
        if make_confusion_matrix:
            cm_savefile = os.path.join("plots", "confusion_matrix.png")
            if shuffle_var != "":
                # Special saveas for
                cm_savefile = os.path.join("plots", "permutation_ranking", f"{shuffle_var}_shuffled_confusion_matrix.png")
            plot_confusion_matrix(y_pred, y_true, prong=self.prong, weights=weights, saveas=cm_savefile)

        self.reset_generator()

        return y_pred, y_true, weights, np.mean(losses), np.mean(accs)

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
        Function that will reset the generator by reseting the DataLoader objects
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
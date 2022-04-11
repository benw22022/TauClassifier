"""
DataWriter Class Definition
____________________________________________________________________
Definition of the DataWriter Class which inherits from DataLoader
This class is useful for dumping NN results to NTuples
This class differs from DataLoader in that it uses uproot.concatenate
instead of uproot.iterate
"""


import ray
import yaml
import uproot
import numpy as np
import awkward as ak
from typing import List, Union, Tuple
from source.dataloader import DataLoader
import tensorflow as tf


class DataWriter(DataLoader):
    
    def __init__(self, file: str, yaml_features_cfg: str) -> None:
        super().__init__((file,), yaml_features_cfg, batch_size=256, step_size='1 GB')

        """
        Instead of loading batches of data just load the full file
        I have the data split up into 100,000 event files to make this easier
        Trying to iterativly fill the result tree is hard
        args:
            files: List[str] - A list of root files to load data from
            yaml_features_cfg: str - A filepath to a yaml config file containing info on input features
        """
        self.big_batch = uproot.concatenate(file, filter_name=self.features)

    def write_results(self, model: tf.keras.Model, output_file: str) -> None:
        """
        Save output and key variables for perf plots to file 
        """
        outfile = uproot.recreate(output_file)

        branch_dict = {}       
        batch, y_true, weights = self.process_batch(self.big_batch)
        y_pred = model.predict(batch)
        branch_dict["TauClassifier_Scores"] = y_pred
        branch_dict["TauClassifier_isFake"] = y_pred[:, 0]
        branch_dict["TauClassifier_is1p0n"] = y_pred[:, 1]
        branch_dict["TauClassifier_is1p1n"] = y_pred[:, 2]
        branch_dict["TauClassifier_is1pXn"] = y_pred[:, 3]
        branch_dict["TauClassifier_is3p0n"] = y_pred[:, 4]
        branch_dict["TauClassifier_is3pXn"] = y_pred[:, 5]
        branch_dict["TauClassifier_TruthScores"] = y_true
        for branch in self.features_config["OutFileBranches"]:
            branch_dict[branch] = self.big_batch[branch]

        branch_dict["TauClassifier_Weight"] = weights

        outfile["tree"] = branch_dict

RayDataWriter = ray.remote(DataWriter)
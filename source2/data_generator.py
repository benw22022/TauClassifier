import ray
import tensorflow as tf
from typing import List, Union, Tuple
import numpy as np
import uproot
import awkward as ak
import yaml
import os
from source2.dataloader import DataLoader


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, tau_files: List[str], jet_files: List[str], yaml_feature_cfg: str, batch_size: int=256):

        self.yaml_feature_config = os.path.abspath(yaml_feature_cfg)
        self.dataloaders = []
        self.batch_size = batch_size
        self.tau_files = tau_files
        self.jet_files = jet_files

        x = uproot.lazy(self.tau_files, filter_name='TauJets_truthDecayMode', libray='np')
        y = uproot.lazy(self.jet_files, filter_name='TauJets_truthDecayMode', libray='np')


        x = x["TauJets_truthDecayMode"]
        y = y["TauJets_truthDecayMode"]
        lx = len(x)
        ly = len(y)
        self.ntaus = lx
        self.njets = ly
        self.nevents = self.ntaus + self.njets

        self.tau_batch_size = int((self.ntaus / self.nevents) * self.batch_size)
        self.jet_batch_size = int((self.njets / self.nevents) * self.batch_size)
        
        self.steps_per_epoch = self.nevents // self.batch_size

        self.tau_loader = DataLoader.remote(self.tau_files, self.yaml_feature_config, self.tau_batch_size)
        self.jet_loader = DataLoader.remote(self.jet_files, self.yaml_feature_config, self.jet_batch_size)

    def __getitem__(self, idx: int):

        # batch = (next(self.tau_loader), next(self.jet_loader) )
        batch = ray.get([self.tau_loader.next.remote(), self.jet_loader.next.remote()])

        x_batch = []
        for k in range(0, len(batch[0][0])):
            x_batch.append(np.concatenate([x[0][k] for x in batch]))

        y_batch = np.concatenate([result[1] for result in batch])
        weight_batch = np.concatenate([result[2] for result in batch])

        return x_batch, y_batch, weight_batch

    def on_epoch_end(self):
        self.tau_loader.terminate.remote()
        self.jet_loader.terminate.remote()
        self.tau_loader = DataLoader.remote(self.tau_files, self.yaml_feature_config, self.tau_batch_size)
        self.jet_loader = DataLoader.remote(self.jet_files, self.yaml_feature_config, self.jet_batch_size)

    def __len__(self):
        return self.steps_per_epoch
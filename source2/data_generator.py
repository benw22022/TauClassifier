import ray
import tensorflow as tf
from typing import List, Union, Tuple
import numpy as np
import uproot
import awkward as ak
import yaml
import os


@ray.remote
class DataLoader:

    def __init__(self, files: List[str], yaml_features_cfg: str, batch_size: int, step_size: Union[str, int]='5 GB'):
        
        self.files = files
        self.batch_size = batch_size
        self.step_size = step_size
   
        with open(yaml_features_cfg, 'r') as stream:
            self.features_config = yaml.load(stream, Loader=yaml.FullLoader)

        self.features = []
        for branch_name in self.features_config["branches"]:
            self.features.extend(self.features_config["branches"][branch_name]["features"])
        self.features.append(self.features_config["Label"])

        self.itr = None
        self.create_itr()
        self.big_batch = next(self.itr)
        self.idx = -1

        histfile = self.features_config["reweight"]["histogram_file"]
        histname = self.features_config["reweight"]["histogram_name"]
        file = uproot.open(histfile)
        self.reweight_hist_edges = file[histname].axis().edges()
        self.reweight_hist_values = file[histname].values()

    def create_itr(self):
        self.itr = uproot.iterate(self.files, filter_name=self.features, step_size=self.step_size)
        self.idx = -1
    
    def terminate(self):
        ray.actor.exit_actor()

    def __next__(self):
        self.idx += 1
        if self.idx * self.batch_size < len(self.big_batch):
            return self.process_batch(self.big_batch[self.idx * self.batch_size: (self.idx + 1) * self.batch_size])
        else:
            try:
                self.big_batch = next(self.itr)
                self.idx = -1
                return self.__next__()
            except StopIteration:
                self.create_itr()
                self.big_batch = next(self.itr)
                return self.__next__()
        
    def next(self):
        return self.__next__()

    def build_array(self, batch, branchname:str, pad_val: int=-999) -> np.ndarray:

        arrays = ak.unzip(batch[self.features_config["branches"][branchname]["features"]])
        max_objs = self.features_config["branches"][branchname]["max_objects"]
        if max_objs > 1:
            arrays = np.stack([ak.to_numpy(ak.pad_none(arr, max_objs, clip=True)) for arr in arrays], axis=1).filled(pad_val) 
            arrays = np.where(abs(arrays) < 1e2, arrays, 0)
            return np.nan_to_num(arrays)
        arrays = np.stack([ak.to_numpy(arr) for arr in arrays], axis=1)
        arrays = np.where(abs(arrays) < 1e2, arrays, 0)
        return np.nan_to_num(arrays)

    def reweight_batch(self, batch) -> np.ndarray:
        reweight_param = self.features_config["reweight"]["feature"]
        return np.asarray(self.reweight_hist_values[np.digitize(batch[reweight_param], self.reweight_hist_edges)])
    
    def process_batch(self, batch) -> Tuple:

        tracks = self.build_array(batch, "TauTracks")
        neutral_pfo = self.build_array(batch,"NeutralPFO")
        shot_pfo = self.build_array(batch,"ShotPFO")
        conv_tracks = self.build_array(batch,"ConvTrack")
        jets = self.build_array(batch,"TauJets")

        labels = np.asarray(batch[self.features_config["Label"]])

        # Only reweight jets
        weights = self.reweight_batch(batch)
        weights = np.where(labels[:, 0] !=0, weights, 1)

        return (tracks, neutral_pfo, shot_pfo, conv_tracks, jets), labels, weights


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
import os
import awkward as ak
import numpy as np
import uproot
from typing import List, Tuple
from source.feature_handler import FeatureHandler
from math import ceil
import yaml

class DataLoader:

    # Helper class to manage just one of each file type
    # This was my original approach - I tried to parallelise the I/O by making DataLoaders run
    # on seperate python processes in parallel using Ray

    def __init__(self, files: List[str], features_config: str, is_tau: bool, batch_size: int=1, stats_csv: str=None) -> None:
        
        self.files = files

        with open(features_config, 'r') as stream:
            self.features_config = yaml.load(stream, Loader=yaml.FullLoader)

        self.batch_size = batch_size

        # This particular version of this class uses uproot.lazy - can also use uproot.iterate (which was my original approach)
        cache = uproot.LRUArrayCache("50 MB")
        self.lazy_array = uproot.lazy(self.files, filter_name=self.features.as_list(), step_size="50 MB", file_handler=uproot.MultithreadedFileSource, array_cache=cache)

        # Load Reweighting Histogram
        self.can_reweight = False
        histfile = self.features_config["reweight"]["histogram_file"]
        histname = self.features_config["reweight"]["histogram_name"]
        try:
            file = uproot.open(histfile)
            try:
                self.reweight_hist_edges = file[histname].axis().edges()
                self.reweight_hist_values = file[histname].values()
                self.can_reweight = True
            except KeyError:
                print(f"Warning: Could not load histogram {histname} from {histfile}. Will not be able to reweight.")
        except FileNotFoundError:
            print(f"Warning: Could not open reweighting file {histfile}. Will not be able to reweight.")

        
    def build_array(self, batch, branchname:str, pad_val: int=0) -> np.ndarray:
        arrays = ak.unzip(batch[self.features_config["branch"][branchname]["features"]])
        max_objs = self.features_config["branch"][branchname]["max_objects"]
        if max_objs > 1:
            return np.stack([ak.to_numpy(ak.pad_none(arr, max_objs, clip=True)) for arr in arrays], axis=1).filled(pad_val)  
        return np.stack([ak.to_numpy(arr) for arr in arrays], axis=1)     


    def __getitem__(self, idx) -> Tuple:

        batch = self.lazy_array[idx * self.batch_size: (idx  + 1) * self.batch_size]
        
        tracks = self.build_array(batch, "TauTracks")
        neutral_pfo = self.build_array(batch,"NeutralPFO")
        shot_pfo = self.build_array(batch,"ShotPFO")
        conv_tracks = self.build_array(batch,"ConvTrack")
        jets = self.build_array(batch,"TauJets")

        if self.is_tau:
            decay_mode = ak.to_numpy(batch[self.features_config["DecayMode"]])
            labels = np.zeros((len(decay_mode), 6))  
            for i, dm in enumerate(decay_mode):
                labels[i][dm + 1] += 1
        else:
            labels = np.zeros((len(jets), 6))
            labels[:, 0] = 1

        weights = self.reweight_batch(batch, dont_rewight=self.is_tau)

        return ((tracks, neutral_pfo, shot_pfo, conv_tracks, jets), labels, weights)

    def reweight_batch(self, batch, dont_reweight: bool=False) -> np.ndarray:
        if self.can_reweight and not dont_reweight:
            reweight_param = self.features_config["reweight"]["feature"]
            return self.reweight_hist_values[np.digitize(batch[reweight_param], self.reweight_hist_dges)]
        else:
            return np.ones((len(batch[reweight_param])))

    def __len__(self) -> int:
        return ceil(len(self.lazy_array) / self.batch_size)

    
import ray
import yaml
import uproot
import awkward as ak
import numpy as np
from typing import List, Tuple
from math import ceil
import pandas as pd

@ray.remote
class DataLoader:

    def __init__(self, files: List[str], yaml_feature_config: str, batch_size: int=1, stats_csv: str=None) -> None:
        
        self.idx = 0
        self.files = files
            
        with open(yaml_feature_config, 'r') as stream:
            self.features_config = yaml.load(stream, Loader=yaml.FullLoader)

        self.batch_size = batch_size
        self.features_list = []
        for branch_name in self.features_config["branches"]:
            self.features_list.extend(self.features_config["branches"][branch_name]["features"])
        self.features_list.append(self.features_config["Label"])

        self.big_batch = uproot.lazy(self.files, filter_name=self.features_list, step_size="5 GB")

        histfile = self.features_config["reweight"]["histogram_file"]
        histname = self.features_config["reweight"]["histogram_name"]
        file = uproot.open(histfile)
        self.reweight_hist_edges = file[histname].axis().edges()
        self.reweight_hist_values = file[histname].values()

    def build_array(self, batch, branchname:str, pad_val: int=-999) -> np.ndarray:

        arrays = ak.unzip(batch[self.features_config["branches"][branchname]["features"]])
        max_objs = self.features_config["branches"][branchname]["max_objects"]
        if max_objs > 1:
            arrays = np.stack([ak.to_numpy(ak.pad_none(arr, max_objs, clip=True)) for arr in arrays], axis=1).filled(pad_val)  
            # arrays = np.where(arrays > 0, np.log10(arrays), 0)

            arrays = np.where(abs(arrays) < 1e2, arrays, 0)
            # try:
            # if np.amax(abs(arrays)) > 100:
            #     print(f"{branchname} {np.amax(arrays)}")
            
            return np.nan_to_num(arrays)
        arrays = np.stack([ak.to_numpy(arr) for arr in arrays], axis=1)
        # arrays = np.where(arrays > 0, np.log10(arrays), 0)

        arrays = np.where(abs(arrays) < 1e2, arrays, 0)
        # if np.amax(abs(arrays)) > 100:
        #     print(f"{branchname} {np.amax(arrays)}")

        return np.nan_to_num(arrays)

    def reweight_batch(self, batch) -> np.ndarray:
        reweight_param = self.features_config["reweight"]["feature"]
        return np.asarray(self.reweight_hist_values[np.digitize(batch[reweight_param], self.reweight_hist_edges)])
    
    def __getitem__(self, idx) -> Tuple:
        
        # Repeat if we run out of data
        if idx * self.batch_size > len(self.big_batch):
            nreps = idx * self.batch_size // len(self.big_batch)
            idx = idx * self.batch_size - nreps * len(self.big_batch)

        start = int(idx * self.batch_size)
        end = int((idx  + 1) * self.batch_size)

        batch = self.big_batch[start: end]

        tracks = self.build_array(batch, "TauTracks")
        neutral_pfo = self.build_array(batch,"NeutralPFO")
        shot_pfo = self.build_array(batch,"ShotPFO")
        conv_tracks = self.build_array(batch,"ConvTrack")
        jets = self.build_array(batch,"TauJets")

        labels = np.asarray(batch[self.features_config["Label"]])

        # Only reweight jets
        weights = self.reweight_batch(batch)
        weights = np.where(labels[:, 0] !=0, weights, 1)

        assert not np.any(np.isnan(tracks))
        assert not np.any(np.isnan(neutral_pfo))
        assert not np.any(np.isnan(shot_pfo))
        assert not np.any(np.isnan(conv_tracks))
        assert not np.any(np.isnan(jets))
        assert not np.any(np.isnan(labels))
        assert not np.any(np.isnan(weights))

        return (tracks, neutral_pfo, shot_pfo, conv_tracks, jets), labels, weights

    def __len__(self) -> int:
        return ceil(len(self.big_batch) / self.batch_size)
    
    def terminate(self) -> None:
        ray.actor.exit_actor()

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size
    
    def make_input(self, batch, branchname: str, pad_val: int=-999):
        features = self.features_config["branches"][branchname]["features"]
        max_objs = self.features_config["branches"][branchname]["max_objects"]

        if max_objs > 1:
            ret_array = np.empty((len(batch), len(features), max_objs))
            for i, f in enumerate(features):
                arr = (batch[f] - self.stats_df.loc[f]["Mean"]) /(self.stats_df.loc[f]["StdDev"] + 1e-8)
                arr = ak.to_numpy(ak.pad_none(arr, max_objs, clip=True)).filled(pad_val)  
                ret_array[:, i, :] = arr
        else:
            ret_array = np.empty((len(batch), len(features)))
            for i, f in enumerate(features):
                arr = (batch[f] - self.stats_df.loc[f]["Mean"]) /(self.stats_df.loc[f]["StdDev"] + 1e-8)
                arr = ak.to_numpy(arr)
                ret_array[:, i] = arr
            
        return ret_array

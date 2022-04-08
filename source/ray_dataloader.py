import os
import awkward as ak
import numpy as np
import uproot
from typing import List, Tuple
from source.feature_handler import FeatureHandler
from math import ceil
import ray
import gc
import yaml

@ray.remote
class DataLoader:

    # Helper class to manage just one of each file type
    # This was my original approach - I tried to parallelise the I/O by making DataLoaders run
    # on seperate python processes in parallel using Ray

    def __init__(self, files: List[str], yaml_feature_config: str, batch_size: int=1) -> None:
        
        self.files = files
        
        with open(yaml_feature_config, 'r') as stream:
            self.features_config = yaml.load(stream, Loader=yaml.FullLoader)

        self.batch_size = batch_size
        self.features_list = []
        for branch_name in self.features_config["branches"]:
            self.features_list.extend(self.features_config["branches"][branch_name]["features"])
        self.features_list.append(self.features_config["Label"])
        self.features_list.append("TauJets_mcEventWeight")

        # This particular version of this class uses uproot.lazy - can also use uproot.iterate (which was my original approach)
        # Also the memory leak here is even worse
        cache = uproot.LRUArrayCache("50 MB") 
        self.lazy_array = uproot.lazy(self.files, filter_name=self.features_list, step_size="50 MB", file_handler=uproot.MultithreadedFileSource, array_cache=cache)
        self.itr = None
        self.create_itr()
        self.big_batch = next(self.itr)
        self.big_batch_pos = 0

    def create_itr(self):
        self.itr = uproot.iterate(self.files, filter_name=self.features_list, step_size='500 MB', file_handler=uproot.MultithreadedFileSource)

    def build_array(self, batch, branchname:str, pad_val: int=0) -> np.ndarray:
        arrays = ak.unzip(batch[self.features_config["branches"][branchname]["features"]])
        max_objs = self.features_config["branches"][branchname]["max_objects"]
        if max_objs > 1:
            return np.stack([ak.to_numpy(ak.pad_none(arr, max_objs, clip=True)) for arr in arrays], axis=1).filled(pad_val)  
        return np.stack([ak.to_numpy(arr) for arr in arrays], axis=1)     

    def __getitem__(self, idx) -> Tuple:

        # I suspect that the leak here might be coming from the line below 
        # I read you can run into memory issues when using slices / views etc - I tried deepcopying the slice but that was
        # painfully slow and didn't completely solve the issue
        # batch = self.lazy_array[idx * self.batch_size: (idx  + 1) * self.batch_size]
        # print(f"Nevents in big_batch = {len(self.big_batch)} \t next big batch after = {ceil(len(self.big_batch) / self.batch_size)}")
        
        if self.big_batch_pos >= ceil(len(self.big_batch) / self.batch_size):
            try:
                self.big_batch = next(self.itr)
            except StopIteration:
                self.create_itr()
                print("Reset itr")
            self.big_batch_pos = 0

        idx = self.big_batch_pos
        start = int(idx * self.batch_size)
        end = int((idx  + 1) * self.batch_size)
        batch = self.big_batch[start: end]
        self.big_batch_pos += 1

        tracks = self.build_array(batch, "TauTracks")
        neutral_pfo = self.build_array(batch,"NeutralPFO")
        shot_pfo = self.build_array(batch,"ShotPFO")
        conv_tracks = self.build_array(batch,"ConvTrack")
        jets = self.build_array(batch,"TauJets")

        labels = batch[self.features_config["Label"]]

        weights = ak.to_numpy(batch["TauJets_mcEventWeight"])

        return ((tracks, neutral_pfo, shot_pfo, conv_tracks, jets), labels, weights)

    def __len__(self) -> int:
        return ceil(len(self.lazy_array) / self.batch_size)
    
    def terminate(self) -> None:
        ray.actor.exit_actor()
    
    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size
        
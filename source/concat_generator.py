import yaml
import uproot
import awkward as ak
import numpy as np
from typing import List, Tuple
import random
from source.logger import logger

def split(a: List, n: int) -> List[List]:
    """
    Simple function to split list a into n equal length parts
    """
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


class DataLoader:

    def __init__(self, files: List[str], batch_size: int) -> None:
        self.batch_size = batch_size
        self.files = files
        self.arrays = [uproot.concatenate(file, filter_name=self.features_list, file_handler=uproot.MultithreadedFileSource) for file in files]
        self.spec_batch_sizes = [len(array) // self.batch_size for array in self.arrays]
    
    def __getitem__(self, idx):
        return [array[idx * bs: (idx + 1 * bs)] for array, bs in zip(self.arrays, self.spec_batch_sizes)]


class DataGenerator:

    def __init__(self, files: List[str], yaml_feature_config: str, batch_size: int=256, max_mem_pct: float=15, nsplits: int=4) -> None:
        
        # Class members
        self.idx = 0
        self.files = files
        self.batch_size = batch_size
        self.max_mem_pct = max_mem_pct

        # Read feature config
        with open(yaml_feature_config, 'r') as stream:
            self.features_config = yaml.load(stream, Loader=yaml.FullLoader)

        # Get list of all input features
        self.features_list = []
        for branch_name in self.features_config["branches"]:
            self.features_list.extend(self.features_config["branches"][branch_name]["features"])
        self.features_list.append(self.features_config["Label"])
        
        # Organise files
        self.nsplits = nsplits
        self.split_index = 0
        random.shuffle(self.files)
        self.file_index = 0
        self.file_list_split = list(split(self.files, self.nsplits))

        # Compute length of dataset
        arr = uproot.concatenate(self.files, filter_name="TauJets_mcEventWeight")
        self.nevents = len(arr)

        # Load reweighting histogram
        histfile = self.features_config["reweight"]["histogram_file"]
        histname = self.features_config["reweight"]["histogram_name"]
        file = uproot.open(histfile)
        self.reweight_hist_edges = file[histname].axis().edges()
        self.reweight_hist_values = file[histname].values()

        # Load initial batch of files
        self.big_batch = self.load_large_batch(self.file_list_split[self.file_index])

    def load_large_batch(self, files: List[str]):
        self.big_batch = [uproot.concatenate(file, filter_name=self.features_list, file_handler=uproot.MultithreadedFileSource) for file in files]
        self.big_batch = random.shuffle(self.big_batch)

    def build_array(self, batch, branchname:str, pad_val: int=-999) -> np.ndarray:

        arrays = ak.unzip(batch[self.features_config["branches"][branchname]["features"]])
        max_objs = self.features_config["branches"][branchname]["max_objects"]
        if max_objs > 1:
            arrays = np.stack([ak.to_numpy(ak.pad_none(arr, max_objs, clip=True)) for arr in arrays], axis=1).filled(pad_val)  

            arrays = np.where(abs(arrays) < 1e2, arrays, 0)
            if np.amax(abs(arrays)) > 100:
                print(f"{branchname} {np.amax(arrays)}")
            
            return np.nan_to_num(arrays)
        arrays = np.stack([ak.to_numpy(arr) for arr in arrays], axis=1)

        arrays = np.where(abs(arrays) < 1e2, arrays, 0)
        if np.amax(abs(arrays)) > 100:
            print(f"{branchname} {np.amax(arrays)}")

        return np.nan_to_num(arrays)

    def reweight_batch(self, batch) -> np.ndarray:
        reweight_param = self.features_config["reweight"]["feature"]
        return np.asarray(self.reweight_hist_values[np.digitize(batch[reweight_param], self.reweight_hist_edges)])
    
    def __getitem__(self, idx) -> Tuple:
        
        self.idx += 1

        # If we reach end of array load next batch
        if self.idx * self.batch_size > len(self.big_batch):
            self.file_index += 1
            self.idx = 0
            self.big_batch = self.load_large_batch(self.file_list_split[self.file_index])

        start = int(self.idx * self.batch_size)
        end = int((self.idx  + 1) * self.batch_size)

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
        return len(self.nevents) // self.batch_size

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

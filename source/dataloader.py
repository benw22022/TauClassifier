"""
DataLoader
______________________________________________________________
Class definition for DataLoader object
A helper class for reading and processing network input data
"""

import logger
log = logger.get_logger(__name__)
import ray
import uproot
import numpy as np
import awkward as ak
from omegaconf import DictConfig
from typing import List, Union, Tuple


class DataLoader:

    def __init__(self, files: List[str], config: DictConfig, batch_size: int=0, step_size: Union[str, int]='5 GB', name: str='DataLoader') -> None:
        """
        Create a new DataLoader for handling input. Instantiated as a ray Actor on new python process
        For efficiency for small batch sizes this code loads a large batch of data and slices of smaller batches for training
        TODO: Add option for cuts (probably have cuts as an option in config file)
        args:
            files: List[str] - A list of root files to load data from
            config: DictConfig - A global config dict from Hydra
            (optoinal) batch_size: int=0 - The batch size for training/inferance. Number of samples yielded when next() is called
            if not set then use maximum possible batch size which is step_size
            (optional) step_size: Union(str, int)='5 GB' - step_size arguement for uproot.iterate. If str then a a memory size e.g. '1 GB' 
            else if an int then the number of samples to load per batch. Rough testing has shown that 5 GB seems to be a good option when running 
            on laptop with 27 GB of available RAM
            (optional) name: str - A optional name to give this DataLoader object 
        """
        self.files = files
        self.batch_size = batch_size
        self.step_size = step_size
        self.config = config
        self.name = name

        # Get a list of all features
        self.features = []
        for branch_name in self.config.branches:
            self.features.extend(self.config.branches[branch_name].features)
        self.features.extend(self.config.OutFileBranches)
        self.features.append(self.config.Label)
        self.features.append(self.config.reweight.feature)

        # Create uproot iterator and load 1st large batch of data
        self.itr = None
        self.create_itr()
        self.big_batch = next(self.itr)
        self.big_batch_idx
        self.idx = -1

        if self.batch_size < 1:
            self.batch_size = len(self.big_batch)

        if len(self.big_batch) < self.batch_size:
            log.warning(f"{self.name} has a batch_size ({self.batch_size}) larger than batch length ({len(self.big_batch)})")
        log.debug(f"{self.name}: Initialized")

    def create_itr(self) -> None:
        """
        Create the iterator
        """
        self.itr = uproot.iterate(self.files, filter_name=self.features, step_size=self.step_size)
        self.idx = -1
        self.big_batch_idx = -1
    
    def terminate(self) -> None:
        """
        Terminate ray actor - useful for freeing memory
        """
        ray.actor.exit_actor()

    def __next__(self) -> Tuple[np.ndarray]:
        """
        Gets next batch of data
        If we run out of data in the currently loaded large batch then move to next 
        If we finish looping through all data then restart the iterator
        """
        self.idx += 1
        if self.idx * self.batch_size < len(self.big_batch):
            log.debug(f"{self.name}: Loading batch {self.idx} from big_batch {self.big_batch_idx}")
            return self.process_batch(self.big_batch[self.idx * self.batch_size: (self.idx + 1) * self.batch_size])
        else:
            try:
                self.big_batch = next(self.itr)
                self.big_batch_idx += 1
                self.idx = -1
                log.debug(f"{self.name}: Loading next big_batch")
                return self.__next__()
            except StopIteration:
                self.create_itr()
                self.big_batch = next(self.itr)
                log.debug(f"{self.name}: Ran out of data - restarting iterator")
                return self.__next__()
        
    def next(self):
        """
        An alias for __next__() - next() on ray actors doesn't work
        """
        return self.__next__()

    def build_array(self, batch: ak.Array, branchname:str, pad_val: int=-999, cutoff: int=1e2) -> np.ndarray:
        """
        Builds input array for input branch from data in feature config
        Pads and clips non-rectilinear arrays
        Note: use ak.fill_none() on the none padded ak arrays. Trying to pad after converting to numpy leads to 
        unitialised memory in the array
        args:
            batch: ak.Array - A batch of data loaded by uproot 
            branchname: Name of NN input branch e.g. TauTracks, NeutralPFO etc... A key in the 'branches' dict in 
            in feature config
            pad_val: The value to pad arrays with N.B. Should be the same as masking value in NN
            cutoff: Maximum absolute value in the output arrays. Useful for removing outliers
        returns:
            np.ndarray - A complete input array for one branch of the network
        """
        
        arrays = ak.unzip(batch[self.config.branches[branchname].features])
        max_objs = self.config.branches[branchname].max_objects

        if max_objs > 1:
            arrays = np.stack([ak.to_numpy(ak.fill_none(ak.pad_none(arr, max_objs, clip=True), pad_val)) for arr in arrays], axis=1)
            return np.nan_to_num(arrays)
        arrays = np.stack([ak.to_numpy(arr) for arr in arrays], axis=1)
        return np.nan_to_num(arrays)

    def process_batch(self, batch: ak.Array) -> Tuple:
        """
        Build input arrays for each branch of the NN, lables and weights
        args:
            batch: ak.Array - Batch of data loaded by uproot
        returns:
            Tuple - A Tuple of arrays for training/inferance. Structure is:
            (x, y, weights) where x = (branch1, branch2, ..., branchn)
        """

        # TODO: can this be done without knowing the number of branches?
        tracks = self.build_array(batch, "TauTracks")
        neutral_pfo = self.build_array(batch,"NeutralPFO")
        shot_pfo = self.build_array(batch,"ShotPFO")
        conv_tracks = self.build_array(batch,"ConvTrack")
        jets = self.build_array(batch,"TauJets")

        labels = ak.to_numpy(batch[self.config.Label])
        weights = ak.to_numpy(batch[self.config.Weight])

        return (tracks, neutral_pfo, shot_pfo, conv_tracks, jets), labels, weights
    
    

RayDataLoader = ray.remote(DataLoader)

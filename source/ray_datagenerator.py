from re import sub
import numpy as np
from typing import List, Dict
from source.feature_handler import FeatureHandler
# from source.ray_dataloader import DataLoader
from source.ray_dataloader_3 import DataLoader
from math import ceil
import math
import ray
import psutil
import os
import random
import tensorflow as tf
import sys

def split(a: List, n: int) -> List[List]:
    """
    Simple function to split list a into n equal length parts
    """
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

class DataGenerator(tf.keras.utils.Sequence):
# class DataGenerator:

    def __init__(self, file_list: List[str], yaml_feature_config: str, batch_size: int=256, max_mem_pct: float=15, ncores: int=2, nsplits: int=1) -> None:
        
        self.dataloaders = []
        self.batch_size = batch_size
        self.file_list = file_list
        self.ncores = ncores
        self.yaml_feature_config = os.path.abspath(yaml_feature_config)

        # Variables to handle excessive memory growth workaround
        self.max_mem_pct = max_mem_pct
        self.nsplits = nsplits
        self.split_index = 0
        random.shuffle(self.file_list)
        self.file_list_split = list(split(self.file_list, self.nsplits))
        self.create_dataloaders()

    def create_dataloaders(self):
        """
        Create Ray Actor DataLoaders
        """

        file_list = self.file_list_split[self.split_index]
        file_list = list(split(file_list, 1))

        for sublist in file_list:
            self.dataloaders.append(DataLoader.remote(sublist, self.yaml_feature_config))
            # self.dataloaders.append(DataLoader(sublist, self.yaml_feature_config))

        self.dataloader_lengths = ray.get([dl.__len__.remote() for dl in self.dataloaders])
        # self.dataloader_lengths = [dl.__len__() for dl in self.dataloaders]
        self.nevents = sum(self.dataloader_lengths)

        for dl in self.dataloaders:
            batch_size = math.ceil(ray.get(dl.__len__.remote()) / self.nevents * self.batch_size)
            dl.set_batch_size.remote(batch_size)
            # batch_size = math.ceil(dl.__len__() / self.nevents * self.batch_size)
            # dl.set_batch_size(batch_size)
                

    def __getitem__(self, idx):
        batch = ray.get([dl.__getitem__.remote(idx) for dl in self.dataloaders])
        # batch = [dl.__getitem__(idx) for dl in self.dataloaders]
        x_batch = []
        for k in range(0, len(batch[0][0])):
            x_batch.append(np.concatenate([x[0][k] for x in batch]))

        y_batch = np.concatenate([result[1] for result in batch])
        weight_batch = np.concatenate([result[2] for result in batch])

        return x_batch, y_batch, weight_batch

    def __len__(self) -> int:
        return ceil(self.nevents / self.batch_size) 

    def reload(self) -> None:
        """
        Kill all DataLoader actors and recreate them
        This is a workaround to fix the evergrowing memory consumption
        """
        print("Reloading DataLoaders")
        # for dl in self.dataloaders:
            # dl.terminate.remote()
        self.dataloaders.clear()
        self.create_dataloaders()
    
    def get_mem_usage(self, verbose=True) -> float:
        """
        Gets current memory usage as a percentage of total system memory
        """
        current_process = psutil.Process(os.getpid())
        mem = current_process.memory_percent()
        for child in current_process.children(recursive=True):
            mem += child.memory_percent()
        if verbose:
            print(f"Memory usage = {mem:2.2f} %")
        return mem


    def on_epoch_end(self) -> None:
        """
        Called when Keras reaches the end of an epoch
        If the memory usage exceeds set threshold then kill all ray dataloader actors 
        and recreate them
        """
        # current_pct_mem = self.get_mem_usage()
        # if current_pct_mem > self.max_mem_pct:
        #     self.reload()
        
        if self.split_index + 1 < len(self.file_list_split) - 1:
            self.split_index += 1
            print(f"Moving to next dataset shard {self.split_index} / {self.nsplits}")
            self.reload()
        else:
            self.split_index = 0
            print(f"Reached end of dataset {self.split_index} / {self.nsplits}")
            self.reload()

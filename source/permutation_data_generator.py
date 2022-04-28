"""
PermuationDataGenerator
__________________________________________________
Generator to perform permuation ranking
"""

import logger
log = logger.get_logger(__name__)
import ray
import numpy as np
from source.data_generator import DataGenerator
from typing import List, Tuple, Union
from omegaconf import DictConfig


class PermutationDataGenerator(DataGenerator):

    def __init__(self, tau_files: List[str], jet_files: List[str], config: DictConfig, batch_size: int = 256, 
                 step_size: Union[str, int] = '1GB', name: str = 'DataGenerator') -> None:
        super().__init__(tau_files, jet_files, config, batch_size, step_size, name)

        self.perm_index = None
        self.rng = np.random.default_rng()

    def __getitem__(self, idx: int) -> Tuple[np.ndarray]:
        """
        Allows DataGenerator to be indexable. Not really though since the index does nothing.
        This method is only provided to satify the requirements for tensorflow generator training.
        This should really be treated as __next__
        args:
            idx: int - Does nothing, just provide it as an arguement so that code works
        returns:
            Tuple[np.ndarrays] - A Tuple of arrays; structure is
            feature arrays, labels, weights
        """

        # batch = (next(self.tau_loader), next(self.jet_loader) )
        batch = ray.get([self.tau_loader.next.remote(), self.jet_loader.next.remote()])

        x_batch = []
        for k in range(0, len(batch[0][0])):
            x_batch.append(np.concatenate([x[0][k] for x in batch]))

        y_batch = np.concatenate([result[1] for result in batch])
        weight_batch = np.concatenate([result[2] for result in batch])

        if self.perm_index is None:
            return x_batch, y_batch, weight_batch

        x_batch[self.perm_index[0]][:, self.perm_index[1]] = self.rng.permutation(x_batch[self.perm_index[0]][:, self.perm_index[1]], axis=0)

        log.info(f"{self.name}: Actual batch size = {len(y_batch)}")

        return x_batch, y_batch, weight_batch
    


    

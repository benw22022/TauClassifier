"""
DataWriter Class Definition
____________________________________________________________________
Definition of the DataWriter Class which inherits from DataLoader
This class is useful for dumping NN results to NTuples
This class differs from DataLoader in that it uses uproot.concatenate
instead of uproot.iterate
"""

import logger
log = logger.get_logger(__name__)
import os
import ray
import tqdm
import uproot
import numpy as np
from source.dataloader import DataLoader
import tensorflow as tf
import matplotlib.pyplot as plt
from omegaconf import DictConfig    

class DataWriter(DataLoader):
    
    def __init__(self, file: str, config: DictConfig) -> None:
        super().__init__((file,), config, batch_size=256, step_size='1 GB')

        """
        Instead of loading batches of data just load the full file
        I have the data split up into 100,000 event files to make this easier
        Trying to iterativly fill the result tree is hard
        args:
            files: List[str] - A list of root files to load data from
            config: DictConfig - A global config dict from Hydra
        """
        self.file = file
        self.big_batch = uproot.concatenate(file, filter_name=self.features)

    def write_results(self, model: tf.keras.Model, output_file: str) -> None:
        """
        Save output and key variables for perf plots to file 
        """
        outfile = uproot.recreate(output_file)

        branch_dict = {}       
        batch, _, _ = self.process_batch(self.big_batch)
        y_pred = model.predict(batch)
        branch_dict["TauClassifier_Scores"] = y_pred
        branch_dict["TauClassifier_isFake"] = y_pred[:, 0]
        branch_dict["TauClassifier_is1p0n"] = y_pred[:, 1]
        branch_dict["TauClassifier_is1p1n"] = y_pred[:, 2]
        branch_dict["TauClassifier_is1pxn"] = y_pred[:, 3]
        branch_dict["TauClassifier_is3p0n"] = y_pred[:, 4]
        branch_dict["TauClassifier_is3pxn"] = y_pred[:, 5]
        for branch in self.config.OutFileBranches:
            try:
                branch_dict[branch] = self.big_batch[branch]
            except ValueError:
                log.warn(f"{self.file} has no branch named {branch}. Is this expected? Filling tree with dummy data")
                branch_dict[branch] = np.ones_like(y_pred[:, 0]) * -999

        outfile["tree"] = branch_dict

    @staticmethod
    def plot_hist(array: np.ndarray, var_name: str, saveas: str) -> None:
        fig, ax = plt.subplots()
        ax.hist(array, bins=500, histtype='step')
        ax.set_xlabel(var_name)
        # ax.set_yscale("symlog")
        # ax.set_xscale("symlog")
        plt.savefig(saveas)

    def make_control_plots(self, outfile_loc: str="control_plots") -> None:
        """
        Make control plots of the output being fed to NN
        """
        # Create save location
        output_dir = os.path.join(outfile_loc, os.path.basename(self.file))
        os.makedirs(output_dir)

        # Load data
        batch, y_true, weights = self.process_batch(self.big_batch)

        nplots = sum([branch_arr.shape[1] for branch_arr in batch]) + 2

        branch_names = self.config.branches.keys()

        with tqdm.tqdm(total=nplots) as pbar:

            for branch_name, branch_arr in zip(branch_names, batch):
                for i in range(0, branch_arr.shape[1]):
                    var_name = self.config.branches[branch_name].features[i]
                    outfile = os.path.join(output_dir, f"{var_name}.png")
                    array  = branch_arr[:, i].ravel()
                    self.plot_hist(array[array != -999], var_name, outfile)
                    # try:
                    #     arr = ak.to_numpy(ak.flatten(self.big_batch[var_name])).ravel()
                    # except Exception:
                    #     arr = ak.to_numpy(self.big_batch[var_name]).ravel()
                    # self.plot_hist(arr, var_name, outfile)
                        
                    pbar.update()
                    plt.close('all')
        
            # Plot labels/weights
            self.plot_hist(y_true.ravel(), "y_true", os.path.join(output_dir, f"y_true.png"))
            pbar.update()

            self.plot_hist(weights, "weights", os.path.join(output_dir, f"weights.png"))
            pbar.update()

RayDataWriter = ray.remote(DataWriter)
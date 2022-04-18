import os
import uproot
import glob
import numpy as np
import pandas as pd
import tqdm
import time
from typing import List

def compute_labels(decay_mode: np.ndarray, is_tau: bool) -> np.ndarray:
    """
    Compute one-hot encoded labels for Tau Classifier. Structure:
    fake = [1, 0, 0, 0, 0 ,0]
    1p0n = [0, 1, 0, 0, 0, 0]
    1p1n = [0, 0, 1, 0, 0, 0]
    1pXn = [0, 0, 0, 1, 0, 0]
    3p0n = [0, 0, 0, 0, 1, 0]
    3pXn = [0, 0, 0, 0, 0, 1]
    args:
        decay_mode: np.ndarray - The truth decay mode of the tau
        is_tau: bool - True if real tau; False if a fake
    returns:
        onehot_labels: np.ndarray - array of one hot encoded labels for classification
    """

    onehot_labels = np.zeros((len(decay_mode), 6))
    if not is_tau:
        onehot_labels[:, 0] = 1
    else:
        for i, dm in enumerate(decay_mode):
            onehot_labels[i][dm + 1] = 1

    return onehot_labels    


def split_files(files: List[str], outpath: str, name: str, is_tau: bool, step_size: int) -> None:
    """
    Splits a group of similar files into smaller chunks of length step_size
    Adds new branches for normalised features (feature - mean / stdDev)
    Adds new branch for one-hot encoded labels
    args:
        files: List[str] - A list of files to be processed
        outpath: str - Output directory to store files
        name: str - A name to give the processed output files
        is_tau: bool - For labels, True if files contain taus, False for fakes
        step_size: int - (Rough) number of events per file
    returns:
        None
    """
    try:
        os.makedirs(outpath)
    except FileExistsError:
        pass
    keys = uproot.open(f"{files[0]}:tree").keys()
    stats_df = pd.read_csv("../TauClassifier2/config/stats_df.csv", index_col=0)
    
    histfilename = "../TauClassifier2/config/ReWeightHist.root"
    histname = "ReWeightHist"
    histfile = uproot.open(histfilename)
    reweight_hist_edges = histfile[histname].axis().edges()
    reweight_hist_values = histfile[histname].values()

    for i, batch in enumerate(uproot.iterate(files, step_size=step_size)):
        i += 1
        start_time = time.time()
        new_file = uproot.recreate(os.path.join(outpath, f"{name}_{i:03d}.root"))

        branch_dict = {}
        for column in keys:
    
            column_fixed = column.replace(".", "_")

            if "TauCluster" in column_fixed:
                continue

            branch_dict[column_fixed] = batch[column]
            try:
                mean = stats_df.loc[column_fixed]["Mean"]
                std = stats_df.loc[column_fixed]["StdDev"]
                branch_dict[f"{column_fixed}_normed"] = (batch[column] - mean) / (std + 1e-8)
            except KeyError:
                pass

        branch_dict["TauClassifier_Labels"] = compute_labels(batch["TauJets.truthDecayMode"], is_tau)
        if is_tau:
            branch_dict["TauClassifier_pTReweight"] = np.ones(len(batch["TauJets.ptJetSeed"]))
        else:
            branch_dict["TauClassifier_pTReweight"] = np.asarray(reweight_hist_values[np.digitize(batch["TauJets.ptJetSeed"], reweight_hist_edges)])
    
        new_file['tree'] = branch_dict
        new_file.close()

        # print(f"Done: {name}_{i:03d}.root in {time.time - start_time}s")

    print(f"All files for {name} done!")

def main() -> None:
    """
    Main code body:
    Splits up files into 100,000 event checks 
    Added extra branches with preprocessing
        - Features are normed by subracting mean and dividing by stdDev
    Adds a new branch for labels
    """
    split_files(glob.glob("../NTuples/*Gammatautau*/*.root"), "../split_NTuples/Gammatautau", "Gammatautau", True, 100000)
    split_files(glob.glob("../NTuples/*JZ1*/*.root"), "../split_NTuples/JZ1", "JZ1", False, 100000)
    split_files(glob.glob("../NTuples/*JZ2*/*.root"), "../split_NTuples/JZ2", "JZ2", False, 100000)
    split_files(glob.glob("../NTuples/*JZ3*/*.root"), "../split_NTuples/JZ3", "JZ3", False, 100000)
    split_files(glob.glob("../NTuples/*JZ4*/*.root"), "../split_NTuples/JZ4", "JZ4", False, 50000)
    split_files(glob.glob("../NTuples/*JZ5*/*.root"), "../split_NTuples/JZ5", "JZ5", False, 100000)
    split_files(glob.glob("../NTuples/*JZ6*/*.root"), "../split_NTuples/JZ6", "JZ6", False, 100000)
    split_files(glob.glob("../NTuples/*JZ7*/*.root"), "../split_NTuples/JZ7", "JZ7", False, 100000)
    split_files(glob.glob("../NTuples/*JZ8*/*.root"), "../split_NTuples/JZ8", "JZ8", False, 100000)
    
if __name__ == "__main__":
    main()
    
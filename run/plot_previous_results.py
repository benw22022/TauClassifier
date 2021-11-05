"""
Script to plot results from current TauID and Decay Mode Classifier
"""

import uproot
import matplotlib.pyplot as plt
import numpy as np
from plotting.plotting_functions import plot_ROC, plot_confusion_matrix
import glob
import os
from config.config import get_cuts
from scripts.utils import logger

def plot_previous():

    JZ_files = glob.glob(os.path.join("../NTuples/*JZ*/*.root"))
    tau_files = glob.glob(os.path.join("../NTuples/*Gammatautau*/*.root"))

    # Compute confusion matrix
    y_true = []
    y_pred = []
    dm_vars = ["TauJets.truthDecayMode", "TauJets.is1p0n", "TauJets.is1p1n", "TauJets.is1pxn", "TauJets.is3p0n", "TauJets.is3pxn"]
    tau_arr = uproot.concatenate(tau_files, cut=get_cuts(), library='np', filter_name=dm_vars)
    
    ntaus = len(tau_arr["TauJets.truthDecayMode"])
    for i, truth_dm in enumerate(tau_arr["TauJets.truthDecayMode"]):
        tmp_arr = [0, 0, 0, 0, 0]
        tmp_arr[truth_dm] = 1
        y_true.append(tmp_arr)
        tmp_arr = [0, 0, 0, 0, 0]
        tmp_arr[0] = tau_arr["TauJets.is1p0n"][i]
        tmp_arr[1] = tau_arr["TauJets.is1p1n"][i]
        tmp_arr[2] = tau_arr["TauJets.is1pxn"][i]
        tmp_arr[3] = tau_arr["TauJets.is3p0n"][i]
        tmp_arr[4] = tau_arr["TauJets.is3pxn"][i]
        y_pred.append(tmp_arr)

        if i % 100000 == 0: 
            logger.log(f"Done {i} events out of {ntaus}")

    y_pred = np.array(y_pred, dtype='float32')
    y_true = np.array(y_true, dtype='float32')

    plot_confusion_matrix(y_pred, y_true, no_jets=True, title=os.path.join("plots", "Bowens_Confusion_Matrix.png"), 
                          saveas=os.path.join("plots", "Bowens_Confusion_Matrix.png"))

    # plot_ROC(y_pred, y_true, weights=)
    
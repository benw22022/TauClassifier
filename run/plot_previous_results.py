"""
Script to plot results from current TauID and Decay Mode Classifier
"""

import uproot
import matplotlib.pyplot as plt
import numpy as np
from plotting.plotting_functions import plot_ROC, plot_confusion_matrix
import glob
import os


if __name__ == "__main__":

    JZ_files = glob.glob(os.path.join("../NTuples/*JZ*/*.root"))
    tau_files = glob.glob(os.path.join("../NTuples/*Gammatautau*/*.root"))

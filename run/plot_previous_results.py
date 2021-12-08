"""
Script to plot results from current TauID and Decay Mode Classifier
"""

import uproot
from tqdm import tqdm
import numpy as np
from plotting.plotting_functions import plot_ROC, plot_confusion_matrix
import glob
import os
from config.config import get_cuts
from source.utils import logger
from source.preprocessing import Reweighter
from config.files import ntuple_dir

def plot_bowens_confusion_matrix(tau_arr):

    y_true = []
    y_pred = []

    for i, truth_dm in enumerate(tqdm(tau_arr["TauJets_truthDecayMode"])):
        tmp_arr = [0, 0, 0, 0, 0, 0]
        tmp_arr[int(truth_dm)+1] = 1
        y_true.append(tmp_arr)
        tmp_arr = [0, 0, 0, 0, 0, 0]
        tmp_arr[0] = 0
        tmp_arr[1] = tau_arr["TauJets.is1p0n"][i]
        tmp_arr[2] = tau_arr["TauJets.is1p1n"][i]
        tmp_arr[3] = tau_arr["TauJets.is1pxn"][i]
        tmp_arr[4] = tau_arr["TauJets.is3p0n"][i]
        tmp_arr[5] = tau_arr["TauJets.is3pxn"][i]
        y_pred.append(tmp_arr)

    y_pred = np.array(y_pred, dtype='float32')
    y_true = np.array(y_true, dtype='float32')

    print(y_true.shape)
    print(y_pred.shape)

    plot_confusion_matrix(y_pred, y_true, title=os.path.join("plots", "Bowens_Confusion_Matrix.png"), 
                          saveas=os.path.join("plots", "Bowens_Confusion_Matrix.png"))

def plot_juans_ROC(tau_arr, jet_arr):

    reweighter = Reweighter(ntuple_dir, prong=None)
    y_true = []
    y_pred = []
    weights = []

    for rnn_pred, jet_pt in tqdm(zip(tau_arr["TauJets_RNNJetScoreSigTrans"], tau_arr["TauJets_ptJetSeed"]), total=len(tau_arr["TauJets.isRNNJetIDLoose"])):
        y_true.append(1)
        y_pred.append(rnn_pred)
        # weights.append(reweighter.reweight(np.array([jet_pt])))

    for rnn_pred, jet_pt in tqdm(zip(jet_arr["TauJets_RNNJetScoreSigTrans"], jet_arr["TauJets_ptJetSeed"]), total=len(jet_arr["TauJets.isRNNJetIDLoose"])):
        y_true.append(0)
        y_pred.append(rnn_pred)
        # weights.append(reweighter.reweight(np.array([jet_pt])))

    plot_ROC(y_true, y_pred, title="Juan's TauID RNN", saveas=os.path.join("plots", "Juans_TauIDRNN_ROC.png"))

def plot_previous():

    JZ_files = glob.glob(os.path.join("../NTuples/*JZ*/*.root"))
    tau_files = glob.glob(os.path.join("../NTuples/*Gammatautau*/*.root"))

    # Compute confusion matrix
    dm_vars = ["TauJets_truthDecayMode", "TauJets_is1p0n", "TauJets_is1p1n", "TauJets_is1pxn", "TauJets_is3p0n", "TauJets_is3pxn"]
    id_vars = ["TauJets_isRNNJetIDLoose", "TauJets_RNNJetScoreSigTrans", "TauJets_ptJetSeed"]
    tau_arr = uproot.concatenate(tau_files, cut=get_cuts(), library='np', filter_name=dm_vars+id_vars)
    jet_arr = uproot.concatenate(JZ_files, cut=get_cuts(), library='np', filter_name=id_vars)

    plot_bowens_confusion_matrix(tau_arr)
    # plot_juans_ROC(tau_arr, jet_arr)
    
    
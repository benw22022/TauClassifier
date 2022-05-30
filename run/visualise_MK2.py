from cProfile import label
from email.mime import base
from tkinter.messagebox import NO

from pytz import utc
import logger
log = logger.get_logger(__name__)
import os
import glob
import uproot
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple, List
from pathlib import Path
from sklearn.metrics import classification_report
from source import plotting_functions as pf
from omegaconf import DictConfig
from hydra.utils import get_original_cwd, to_absolute_path
import awkward as ak
import sys
from enum import Enum


def get_results_dir(config: DictConfig) -> str:
    """
    Get path to ntuples from given results folder or get last created results folder
    args:
        config: DictConf - Hydra config object
    returns:
        results_dir: str - Path to directory containing evaluated ntuples
    """
    try:
        results_dir = os.path.join(get_original_cwd(), config.results)
        log.info(f"Loading evaluated ntuples from specified file: {results_dir}")
    except AttributeError:
        avail_results_dir = glob.glob(os.path.join(get_original_cwd(), "outputs", "train_output", "*", "results"))
        results_dir = max(avail_results_dir, key=os.path.getctime)
        log.info(f"Loading last created evaluated ntuples: {results_dir}")
    return results_dir


class NTupleReader:
    
    def __init__(self, config: DictConfig, results_dir: str, cut:str=None) -> None:
        self.config = config
        self.results_dir = results_dir
        self.cut = cut
        self.data = None
        self.load_data()
        
    def load_data(self, new_cut: str=None) -> None:

        tau_ntuples = glob.glob(os.path.join(self.results_dir, '*tau*.root'))
        jet_ntuples = glob.glob(os.path.join(self.results_dir, '*jet*.root')) 
        cuts = []
        
        if self.cut is not None:
            cuts.append(self.cut)
        if new_cut is not None:
            cuts.append(new_cut)
        
        cut_str = None
        if len(cuts) > 0:
            cut_str = f"{cuts[0]}"
            if len(cuts) > 1:
                for i in range(len(cuts), 1):
                    cut_str += f"& {cuts[i]}"
            
        tau_data = uproot.concatenate(tau_ntuples, cut=cut_str, library='ak')
        jet_data = uproot.concatenate(jet_ntuples, cut=cut_str, library='ak')    
        
        self.data = ak.concatenate((tau_data, jet_data))
    
    def change_basecut(self, cut):
        self.cut = cut 
        self.load_data()
        
    def __getitem__(self, key: str) -> np.ndarray:
        return ak.to_numpy(self.data[key])

    @property
    def y_true(self) -> np.ndarray:
        return ak.to_numpy(self.data["TauClassifier_Labels"])
    
    @property 
    def y_pred(self) -> np.ndarray:
        return ak.to_numpy(self.data["TauClassifier_Scores"])
    
    @property 
    def weight(self) -> np.ndarray:
        return ak.to_numpy(self.data["TauClassifier_pTReweight"])
    
    @property
    def y_pred_prev(self) -> np.ndarray:
        combined_scores = np.column_stack([
                        1 - ak.to_numpy(self.data["TauJets_RNNJetScoreSigTrans"]),
                        ak.to_numpy(self.data["TauJets_is1p0n"]),
                        ak.to_numpy(self.data["TauJets_is1p1n"]),
                        ak.to_numpy(self.data["TauJets_is1pxn"]),
                        ak.to_numpy(self.data["TauJets_is3p0n"]),
                        ak.to_numpy(self.data["TauJets_is3pxn"])])
        return combined_scores


# TODO: include pT reweighting?
def get_tauid_efficiency_wp_cut(y_true: np.ndarray, y_pred: np.ndarray, wp_eff: int) -> str:
    correct_pred_taus = y_pred[y_true == 1]
    cut_val = np.percentile(correct_pred_taus, 100 - wp_eff)
    return cut_val

def get_tauid_rejection_wp_cut(y_true: np.ndarray, y_pred: np.ndarray, wp_rej: int) -> str:
    correct_pred_fakes = y_pred[y_true == 0]
    cut_val = np.percentile(correct_pred_fakes, wp_rej)
    return cut_val


def make_perfplots(config: DictConfig, wp_type: str="baseline", working_point: int=None, wp_bins=None) -> None:

    # Parse Arguements
    wp_dir = f"{wp_type}"
    if working_point is not None:
        wp_dir = f"{wp_type}_{working_point}"
    utc_cut = None
    rnn_cut = None
    
    # Make plotting directory
    results_dir = get_results_dir(config)
    run_dir = Path(results_dir).parents[0]
    plotting_dir = os.path.join(run_dir, "plots", wp_dir)
    log.error(f"plotting dir = {plotting_dir}")
    os.makedirs(plotting_dir, exist_ok=True)
    
    # Load data
    utc_data = NTupleReader(config, results_dir, cut=utc_cut)
    rnn_data = NTupleReader(config, results_dir, cut=rnn_cut)
    
    if working_point is not None:
        
        if 0 <= working_point <= 100: 
            pass
        else:
            log.error(f"Working point must be between 0 and 100! You asked for {wp_type} {working_point}%")
            raise ValueError
        
        # TODO: would enums be better for this or is that overkill?
        if wp_type == "efficiency":
            utc_cut_val = get_tauid_efficiency_wp_cut(1 - utc_data.y_true[:, 0], 1 - utc_data.y_pred[:, 0], working_point)
            rnn_cut_val = get_tauid_efficiency_wp_cut(1 - rnn_data.y_true[:, 0], 1 - rnn_data.y_pred_prev[:, 0], working_point)
            log.info(f"@{working_point}% Efficiency UTC Score Cut = {utc_cut_val}")
            log.info(f"@{working_point}% Efficiency RNN Score Cut = {rnn_cut_val}")
            utc_cut = f"TauClassifier_isTau > {utc_cut_val}"
            rnn_cut = f"TauJets_RNNJetScoreSigTrans > {rnn_cut_val}"
            utc_data.change_basecut(f"TauClassifier_isTau > {utc_cut_val}")
            rnn_data.change_basecut(f"TauJets_RNNJetScoreSigTrans > {rnn_cut_val}")
        
        elif wp_type == "rejection":
            utc_cut_val = get_tauid_rejection_wp_cut(1 - utc_data.y_true[:, 0], 1 - utc_data.y_pred[:, 0], working_point)
            rnn_cut_val = get_tauid_rejection_wp_cut(1 - rnn_data.y_true[:, 0], 1 - rnn_data.y_pred_prev[:, 0], working_point)
            log.info(f"@{working_point}% Rejection UTC Score Cut = {utc_cut_val}")
            log.info(f"@{working_point}% Rejection RNN Score Cut = {rnn_cut_val}")
            utc_cut = f"TauClassifier_isTau > {utc_cut_val}"
            rnn_cut = f"TauJets_RNNJetScoreSigTrans > {rnn_cut_val}"    
            utc_data.change_basecut(f"TauClassifier_isTau > {utc_cut_val}")
            rnn_data.change_basecut(f"TauJets_RNNJetScoreSigTrans > {rnn_cut_val}")
        
        else:
            log.error(f"You can have either efficiency or rejection working points not {wp_type}")
            raise ValueError
   
    # Print some metrics
    labels = ['fakes', '1p0n', '1p1n', '1pxn', '3p0n', '3pxn']
    report = classification_report(np.argmax(utc_data.y_pred, axis=1), np.argmax(utc_data.y_true, axis=1), target_names=labels)
    log.info(f"Metrics for UTC {wp_dir} \n {report}")

    report = classification_report(np.argmax(rnn_data.y_pred_prev, axis=1), np.argmax(rnn_data.y_true, axis=1), target_names=labels)
    log.info(f"Metrics for RNN {wp_dir} \n {report}")

    
    # Plot some ROC Curves
    _, ax = pf.create_ROC_plot_template()
    ax.plot(*pf.get_efficiency_and_rejection(1 - utc_data.y_true[:,0], 1- utc_data.y_pred[:,0], utc_data.weight), label='UTC: all prong')
    ax.plot(*pf.get_efficiency_and_rejection(1 - rnn_data.y_true[:,0], 1 - rnn_data.y_pred_prev[:,0], rnn_data.weight), label='TauIDRNN all prong')
    ax.legend(title=wp_dir)
    plot_name = "ROC_comp.png"
    ax.set_title(f"plots/{wp_dir}/{plot_name}", loc='right', fontsize=5)
    saveas = os.path.join(plotting_dir, plot_name)
    plt.savefig(saveas, dpi=300)
    log.info(f"Plotted: {saveas}")
    
    
    # UTC ROC Prongs:
    _, ax = pf.create_ROC_plot_template()
    ax.plot(*pf.get_efficiency_and_rejection(1 - utc_data.y_true[:,0], 1 - utc_data.y_pred[:,0], utc_data.weight), label='UTC: all prong')
    utc_data.load_data("TauJets_truthProng == 1")
    ax.plot(*pf.get_efficiency_and_rejection(1 - utc_data.y_true[:,0], 1 - utc_data.y_pred[:,0], utc_data.weight), label='UTC: 1 prong')
    utc_data.load_data("TauJets_truthProng == 3")
    ax.plot(*pf.get_efficiency_and_rejection(1 - utc_data.y_true[:,0], 1 - utc_data.y_pred[:,0], utc_data.weight), label='UTC: 3 prong')
    utc_data.load_data()
    ax.legend(title=wp_dir)
    plot_name = "ROC_utc.png"
    ax.set_title(f"plots/{wp_dir}/{plot_name}", loc='right', fontsize=5)
    saveas = os.path.join(plotting_dir, plot_name)
    plt.savefig(saveas, dpi=300)
    log.info(f"Plotted: {saveas}")
    
    
    # TauID RNN ROC Prongs:
    _, ax = pf.create_ROC_plot_template()
    ax.plot(*pf.get_efficiency_and_rejection(1 - rnn_data.y_true[:,0], 1 - rnn_data.y_pred_prev[:,0], rnn_data.weight), label='TauIDRNN: all prong')
    rnn_data.load_data("TauJets_truthProng == 1")
    ax.plot(*pf.get_efficiency_and_rejection(1 - rnn_data.y_true[:,0], 1 - rnn_data.y_pred_prev[:,0], rnn_data.weight), label='TauIDRNN: 1 prong')
    rnn_data.load_data("TauJets_truthProng == 3")
    ax.plot(*pf.get_efficiency_and_rejection(1 - rnn_data.y_true[:,0], 1 - rnn_data.y_pred_prev[:,0], rnn_data.weight), label='TauIDRNN: 3 prong')
    rnn_data.load_data()
    ax.legend(title=wp_dir)
    plot_name = "ROC_rnn.png"
    ax.set_title(f"plots/{wp_dir}/{plot_name}", loc='right', fontsize=5)
    saveas = os.path.join(plotting_dir, plot_name)
    plt.savefig(saveas, dpi=300)
    log.info(f"Plotted: {saveas}")
    
    # Plot Confusion Matrices
    plot_name = "confusion_matrix_utc.png"
    plot_title = f"plots/{wp_dir}/{plot_name}"
    saveas=os.path.join(plotting_dir, plot_name)
    pf.plot_confusion_matrix(utc_data.y_true, utc_data.y_pred, weights=utc_data.weight, title=plot_title, saveas=saveas)
    log.info(f"Plotted: {saveas}")
    
    plot_name = "confusion_matrix_prev.png"
    plot_title = f"plots/{wp_dir}/{plot_name}"
    saveas=os.path.join(plotting_dir, plot_name)
    pf.plot_confusion_matrix(utc_data.y_true, utc_data.y_pred_prev, weights=utc_data.weight, title=plot_title, saveas=saveas)
    log.info(f"Plotted: {saveas}")

    # Plot Network output
    _, ax= plt.subplots()
    
    ax.hist(1 - utc_data.y_pred[:, 0][utc_data["TauClassifier_isTrueTau"] == 1], bins=100, label='taus UTC', alpha=0.4, color='orange' )
    ax.hist(1 - utc_data.y_pred[:, 0][utc_data["TauClassifier_isTrueFake"] == 1], bins=100, label='fakes UTC', alpha=0.4, color='blue')
    ax.hist(1 - rnn_data.y_pred_prev[:, 0][rnn_data["TauClassifier_isTrueTau"] == 1], bins=100, label='taus RNN', histtype='step', color='orange')
    ax.hist(1 - rnn_data.y_pred_prev[:, 0][rnn_data["TauClassifier_isTrueFake"] == 1], bins=100, label='fakes RNN', histtype='step', color='blue')
    ax.legend()
    ax.set_yscale("log")
    # ax.set_xscale("log")#
    plot_name = "network_output.png"
    ax.set_title(f"plots/{wp_dir}/{plot_name}", loc='right', fontsize=5)
    saveas = os.path.join(plotting_dir, plot_name)
    plt.savefig(saveas, dpi=300)
    log.info(f"Plotted {saveas}")
    
    # Make histograms
    hist_dict = {"TauJets_mu": None, "TauJets_ptJetSeed": None, "TauJets_etaJetSeed": None, "TauJets_phiJetSeed": None}
    
    histograms = {'utc': hist_dict, 'rnn': hist_dict}
    
    for key in hist_dict:
        if wp_bins is None:
            bins = 25
        else:
            bins = wp_bins[key]
        
        log.warn(f"Bins = {bins} - wp = {working_point}")
        
        histograms['utc'][key] = np.histogram(utc_data[key], bins=bins)
        histograms['rnn'][key] = np.histogram(rnn_data[key], bins=bins)

    return histograms
        
        
        

def visualise(config: DictConfig):

    histograms = {}
    
    histograms["baseline"] = make_perfplots(config)
    
    h_bins = {}
    for key in histograms['baseline']['utc']:
        h_bins[key] = histograms['baseline']['utc'][key][0]
    
    # Efficiency working points
    for wp in config.working_points:
        histograms[f"eff_{wp}"] = make_perfplots(config, "efficiency", wp, wp_bins=h_bins)
        
    # Rejction working points
    for wp in config.rejection_working_points:
        histograms[f"rej_{wp}"] = make_perfplots(config, "rejection", wp, wp_bins=h_bins)
    
    # Make efficiency/rejection plots
    results_dir = get_results_dir(config)
    run_dir = Path(results_dir).parents[0]
    plotting_dir = os.path.join(run_dir, "plots_mk2")
    
    # Make phiJetSeed plot
    _, ax = plt.subplots()
    bins, baseline = histograms['baseline']['utc']["TauJets_phiJetSeed"]
    _, eff_95 = histograms['eff_95']['utc']["TauJets_phiJetSeed"]
    _, eff_75 = histograms['eff_75']['utc']["TauJets_phiJetSeed"]
    _, eff_60 = histograms['eff_60']['utc']["TauJets_phiJetSeed"]
    
    bincentres = [(bins[i]+bins[i+1])/2. for i in range(len(bins)-1)]
    log.warn(len(eff_95))
    log.warn(len(h_bins["TauJets_phiJetSeed"]))
    log.warn(len(baseline))
    log.warn(len(bins))
    log.warn(eff_95 / baseline)
    
    ax.step(bincentres, eff_95 / baseline ,where='mid', label=f'WP = 95')
    ax.step(bincentres, eff_75 / baseline ,where='mid', label=f'WP = 75')               
    ax.step(bincentres, eff_60 / baseline ,where='mid', label=f'WP = 60')                              
    ax.set_xlabel("TauJets_phiJetSeed")
    ax.set_ylabel("Efficiency")
    ax.legend(title='UTC')
    plot_name = "network_output.png"
    ax.set_title(f"plots/{plot_name}", loc='right', fontsize=5)
    saveas = os.path.join(plotting_dir, plot_name)
    plt.savefig(saveas, dpi=300)
    log.info(f"Plotted {saveas}")
    
            
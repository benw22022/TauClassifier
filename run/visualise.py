"""
Visualise
_______________________________________________
Make performance plots
"""

import logger
log = logger.get_logger(__name__)
import os
import glob
import uproot
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List
from pathlib import Path
from source import plotting_functions as pf
from omegaconf import DictConfig
from hydra.utils import get_original_cwd, to_absolute_path
import awkward as ak

# TODO: Add this to config
WORKING_POINT_EFFS = (95, 75, 50, 40, 30, 20, 10)

def get_UTC_scores(results_dir: str, cut: str=None) -> Tuple[np.ndarray]:
    """
    Load y_true, y_pred and weights from directory contatining evaluated NTuples
    """
    data = uproot.concatenate(glob.glob(os.path.join(results_dir, "*.root")), library='np', cut=cut)
    y_true = data["TauClassifier_Labels"]
    y_pred = data["TauClassifier_Scores"]
    weights = data["TauClassifier_pTReweight"]
    return y_true, y_pred, weights

def get_TauIDRNN_scores(results_dir: str, cut: str=None) -> Tuple[np.ndarray]:
    """
    Load y_true, y_pred and weights from directory contatining evaluated NTuples
    """
    data = uproot.concatenate(glob.glob(os.path.join(results_dir, "*.root")), library='np', cut=cut)
    y_true = 1 - data["TauClassifier_Labels"][:, 0]
    y_pred = data["TauJets_RNNJetScoreSigTrans"]
    weights = data["TauClassifier_pTReweight"]
    return y_true, y_pred, weights


def get_tauid_wp_cut(y_true: np.ndarray, y_pred: np.ndarray, wp_effs: List[int]) -> List[float]:
    """
    Get list of cuts for specific working point efficiency
    """
    cuts = []
    for wp in wp_effs:
        correct_pred_taus = y_pred[y_true == 1]
        print(correct_pred_taus)
        cuts.append(np.percentile(correct_pred_taus, wp))
    return cuts

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


def create_ROC_plot_template(name: str='ROC'):
    """
    Create a template matplotlib figure for plotting ROC curves
    """
    fig, ax = plt.subplots()
    ax.set_xlabel('Signal Efficiency')
    ax.set_ylabel('Background Rejection')
    ax.set_ylim(1e0, 1e4)
    ax.set_yscale("log")
    ax.set_title(f"plots/{name}.png", loc='right', fontsize=5)
    return fig, ax


class ResultLoader:

    def __init__(self, ntuples: List[str], cuts: str=None, descr: str=''):
        self.ntuples = ntuples
        self.cuts = cuts
        self.descr = descr
        self.data = uproot.lazy(self.ntuples, cut=self.cuts)
        self.wp_cuts = get_tauid_wp_cut(1 - self.y_true[:, 0], 1 - self.y_pred[:, 0], WORKING_POINT_EFFS)
    
    def change_cuts(self, cuts: str):
        self.cuts = cuts
        self.data = uproot.lazy(self.ntuples, cut=self.cuts)

    @property
    def y_true(self):
        return ak.to_numpy(self.data["TauClassifier_Labels"])

    @property
    def y_pred(self):
        return ak.to_numpy(self.data["TauClassifier_Scores"])
    
    @property
    def weights(self):
        self._weights = ak.to_numpy(self.data["TauClassifier_pTReweight"])    
    
    def get_eff_rej(self):
        self.eff, self.allp = pf.get_efficiency_and_rejection(self.y_true[:, 0], self.y_pred[:, 0], self.weights)


def visualise(config: DictConfig) -> None:
    
    log.info("Runing visualise")

    # Remove matplotlib popups
    matplotlib.use('Agg')

    # Set up directories
    results_dir = get_results_dir(config)
    ntuples = glob.glob(os.path.join(results_dir, "*.root"))
    run_dir = Path(results_dir).parents[0]
    plotting_dir = os.path.join(run_dir, "plots")
    os.makedirs(plotting_dir, exist_ok=True)

    # Grab data
    y_true, y_pred, weights = get_UTC_scores(results_dir)
    y_true_1p, y_pred_1p, weights_1p = get_UTC_scores(results_dir, cut="TauJets_truthProng == 1")
    y_true_3p, y_pred_3p, weights_3p = get_UTC_scores(results_dir, cut="TauJets_truthProng == 3")
    y_true_RNN, y_pred_RNN, weights_RNN = get_TauIDRNN_scores(results_dir)


    # Plot output
    fig, ax= plt.subplots()
    yp = 1 - y_pred[:, 0 ]
    yt = 1 - y_true[:, 0 ]
    bins = 100
    ax.hist(yp[yt == 1], bins=100, label='taus UTC', alpha=0.4, color='orange' )
    ax.hist(yp[yt == 0], bins=100, label='fakes UTC', alpha=0.4, color='blue')
    ax.hist(y_pred_RNN[y_true_RNN == 1], bins=100, label='taus RNN', histtype='step', color='orange')
    ax.hist(y_pred_RNN[y_true_RNN == 0], bins=100, label='fakes RNN', histtype='step', color='blue')
    ax.legend()
    ax.set_yscale("log")
    ax.set_xscale("log")
    plt.savefig(os.path.join(plotting_dir, "NN_output.png"), dpi=300)
    
    # Basic plots 
    # TODO: Add plots to be made to config
    pf.plot_ROC(y_true[:, 0], y_pred[:, 0], weights, saveas=os.path.join(plotting_dir, "ROC_allProng.png"))
    pf.plot_confusion_matrix(y_true, y_pred, weights=weights, saveas=os.path.join(plotting_dir, "conf_matrix_allProng.png"))

    # ROC for each prong
    eff_allp, rej_allp = pf.get_efficiency_and_rejection(y_true[:, 0], y_pred[:, 0], weights)
    eff_1p, rej_1p = pf.get_efficiency_and_rejection(y_true_1p[:, 0], y_pred_1p[:, 0], weights_1p)
    eff_3p, rej_3p = pf.get_efficiency_and_rejection(y_true_3p[:, 0], y_pred_3p[:, 0], weights_3p)
    _, ax = create_ROC_plot_template("ROC_prong.png")
    ax.plot(eff_1p, rej_1p, label="ROC 1-prong")
    ax.plot(eff_3p, rej_3p, label="ROC 3-prong")
    ax.plot(eff_allp, rej_allp, label="ROC Combined")
    ax.legend()
    plt.savefig(os.path.join(plotting_dir, "ROC_prong.png"))


    # TauID RNN and UTC before cuts
    eff_allp, rej_allp = pf.get_efficiency_and_rejection(y_true[:, 0], y_pred[:, 0], weights)
    eff_RNN, rej_RNN = pf.get_efficiency_and_rejection(y_true_RNN, y_pred_RNN, weights_RNN)
    _, ax = create_ROC_plot_template("ROC_prong.png")
    ax.plot(eff_allp, rej_allp, label="ROC UTC")
    ax.plot(eff_RNN, rej_RNN, label="ROC TauID RNN")
    ax.legend()
    plt.savefig(os.path.join(plotting_dir, "ROC_TauIDRNN.png"))


    """
    Working points
    """
    wp_cuts = get_tauid_wp_cut(1 - y_true[:, 0], 1 - y_pred[:, 0], WORKING_POINT_EFFS)

    y_true_RNN, y_pred_RNN, weights_RNN = get_TauIDRNN_scores(results_dir)
    tauid_rnn_wp_cuts = get_tauid_wp_cut(y_true_RNN, y_pred_RNN, WORKING_POINT_EFFS)

    for wp_eff, wp_cut_utc, wp_cut_rnn in zip(WORKING_POINT_EFFS, wp_cuts, tauid_rnn_wp_cuts): 

        log.info(f"WP: {wp_eff}\t UTC cut = {wp_cut_utc}\t RNN cut = {wp_cut_rnn}")

        y_true_UTC, y_pred_UTC, weights_UTC = get_UTC_scores(results_dir, cut=f"TauClassifier_isFake < {wp_cut_utc}")
        y_true_RNN, y_pred_RNN, weights_RNN = get_TauIDRNN_scores(results_dir, cut=f"TauJets_RNNJetScoreSigTrans > {wp_cut_rnn}")
        y_true_UTC_withRNN, y_pred_UTC_withRNN, weights_UTC_withRNN = get_UTC_scores(results_dir, cut=f"TauJets_RNNJetScoreSigTrans > {wp_cut_rnn}")

        # ROC comparing with TauID RNN
        eff_RNN, rej_RNN = pf.get_efficiency_and_rejection(y_true_RNN, y_pred_RNN, weights_RNN)
        eff_UTC, rej_UTC = pf.get_efficiency_and_rejection(y_true_UTC[:, 0], y_pred_UTC[:, 0], weights_UTC)
        _, ax = create_ROC_plot_template("ROC_prong.png")
        ax.plot(eff_RNN, rej_RNN, label=f"ROC TauID RNN @ {wp_eff} eff")
        ax.plot(eff_UTC, rej_UTC, label=f"ROC UTC @ {wp_eff} eff")
        ax.legend()
        plt.savefig(os.path.join(plotting_dir, f"ROC_{wp_eff}_comp.png"))
        
        # Plot confusion matrix with TauID cuts - drop the isFake column from y_true & y_pred
        labels = ["1p0n", "1p1n", "1pxn", "3p0n", "3pxn"]
        y_true_UTC_withRNN = np.delete(y_true_UTC_withRNN, 0, axis=1)
        y_pred_UTC_withRNN = np.delete(y_pred_UTC_withRNN, 0, axis=1)
        y_true_UTC = np.delete(y_true_UTC, 0, axis=1)
        y_pred_UTC = np.delete(y_pred_UTC, 0, axis=1)
        
        pf.plot_confusion_matrix(y_true_UTC_withRNN, y_pred_UTC_withRNN, weights_UTC_withRNN, saveas=os.path.join(plotting_dir, f"conf_matrix_TauIDRNN_{wp_eff}.png"), labels=labels)
        pf.plot_confusion_matrix(y_true_UTC, y_pred_UTC, weights=weights_UTC, saveas=os.path.join(plotting_dir, f"conf_matrix_TauIDUTC_{wp_eff}.png"), labels=labels)


if __name__ == "__main__":
    visualise()
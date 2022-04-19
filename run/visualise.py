import glob
import uproot
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List
from source import plotting_functions as pf
from omegaconf import DictConfig

WORKING_POINT_EFFS = (95, 75, 50)

def get_scores(cut: str=None):
    data = uproot.concatenate(glob.glob("results/*.root"), library='np', cut=cut)
    y_true = data["TauClassifier_TruthScores"]
    y_pred = data["TauClassifier_Scores"]
    weights = data["TauClassifier_Weight"]
    return y_true, y_pred, weights

def get_tauid_wp_cut(y_true: np.ndarray, y_pred: np.ndarray, wp_effs: List[int]) -> List[float]:
    cuts = []
    for wp in wp_effs:
        correct_pred_taus = y_pred[y_true == 0]
        cuts.append(np.percentile(correct_pred_taus, wp))
    return cuts

def visualise(config: DictConfig) -> None:
    
    # Remove matplotlib popups
    matplotlib.use('Agg')

    # Grab data
    y_true, y_pred, weights = get_scores()
    y_true_1p, y_pred_1p, weights_1p = get_scores(cut="TauJets_truthProng == 1")
    y_true_3p, y_pred_3p, weights_3p = get_scores(cut="TauJets_truthProng == 3")

    # Basic plots
    pf.plot_ROC(y_true[:, 0], y_pred[:, 0], weights, saveas="plots/ROC_allProng.png")

    # ROC for each prong
    eff_allp, rej_allp = pf.get_efficiency_and_rejection(y_true[:, 0], y_pred[:, 0], weights)
    eff_1p, rej_1p = pf.get_efficiency_and_rejection(y_true_1p[:, 0], y_pred_1p[:, 0], weights_1p)
    eff_3p, rej_3p = pf.get_efficiency_and_rejection(y_true_3p[:, 0], y_pred_3p[:, 0], weights_3p)
    fig, ax = plt.subplots()
    ax.plot(eff_1p, rej_1p, label="ROC 1-prong")
    ax.plot(eff_3p, rej_3p, label="ROC 3-prong")
    ax.plot(eff_allp, rej_allp, label="ROC Combined")
    ax.set_xlabel('Signal Efficiency')
    ax.set_ylabel('Background Rejection')
    ax.set_ylim(1e0, 1e4)
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("plots/ROC_prong.png", loc='right', fontsize=5)
    plt.savefig("plots/ROC_prong.png")

    wp_cuts = get_tauid_wp_cut(y_true[:, 0], y_pred[:, ], WORKING_POINT_EFFS)
    y_true_95RNN, y_pred_95RNN, weights_95RNN = get_scores(cut="TauJets_RNNJetScoreSigTrans > 0.95")
    y_true_95UTC, y_pred_95UTC, weights_95UTC = get_scores(cut="TauClassifier_isFake < 0.95")
    
    pf.plot_confusion_matrix(y_true_95RNN, y_pred_95RNN, weights=weights_95RNN, no_jets=True, saveas="plots/conf_matrix_TauIDRNN95.png")
    pf.plot_confusion_matrix(y_true_95UTC, y_pred_95UTC, weights=weights_95UTC, no_jets=True, saveas="plots/conf_matrix_TauIDUTC95.png")
    pf.plot_confusion_matrix(y_true, y_pred, weights=weights, saveas="plots/conf_matrix_allProng.png")
    # pf.plot_confusion_matrix(y_true, y_pred, weights=weights, saveas="plots/conf_matrix_allProng.png")


if __name__ == "__main__":
    visualise()
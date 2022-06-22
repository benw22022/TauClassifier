"""
Make performance plots
_____________________________________________________________
TODO: This is horrible make this better!
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
from sklearn.metrics import classification_report
from source import plotting_functions as pf
from omegaconf import DictConfig
from hydra.utils import get_original_cwd, to_absolute_path
import awkward as ak
import sys

def get_tauid_wp_cut(y_true: np.ndarray, y_pred: np.ndarray, wp_effs: List[int]) -> List[float]:
    """
    Get list of cuts for specific working point efficiency
    """
    cuts = []
    for wp in wp_effs:
        correct_pred_taus = y_pred[y_true == 1]
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

class ResultLoader:
    # TODO: add global documentation
    # TODO: Should probably move this into source?
    def __init__(self, config: DictConfig, result_type: str, cuts: str=None):
        self.type = result_type
        self.config = config
        # TODO: Work out how to apply the global cuts to this
        #? Try this? 
        self.results_dir = get_results_dir(config)
        self.tau_ntuples = glob.glob(os.path.join(self.results_dir, '*tau*.root'))
        self.jet_ntuples = glob.glob(os.path.join(self.results_dir, '*jet*.root'))  
        self.cuts = cuts
        self.tau_cuts = config.tau_cuts
        self.jet_cuts =  config.fake_cuts 
        
        self.data = self.get_data()
        
        self._y_true = ak.to_numpy(self.data[self.config.visualiser[self.type].y_true])
        self._y_pred = ak.to_numpy(self.data[self.config.visualiser[self.type].y_pred])
        self._weights = ak.to_numpy(self.data[self.config.visualiser[self.type].weights])
    
    def change_cuts(self, cuts: str):
        self.cuts = cuts
        self.data = self.get_data()
    
    @property
    def y_true(self):
        return ak.to_numpy(self.data[self.config.visualiser[self.type].y_true])

    @property
    def y_pred(self):
        return ak.to_numpy(self.data[self.config.visualiser[self.type].y_pred])
    
    @property
    def weights(self):
        return ak.to_numpy(self.data[self.config.visualiser[self.type].weights])
    
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

    
    
    def get_data(self):
        if self.cuts is None:
            tau_data = uproot.concatenate(self.tau_ntuples, cut=self.tau_cuts, library='ak')
            jet_data = uproot.concatenate(self.jet_ntuples, cut=self.jet_cuts, library='ak')    
        else:
            tau_data = uproot.concatenate(self.tau_ntuples, cut=f"({self.cuts}) & ({self.tau_cuts})", library='ak')
            jet_data = uproot.concatenate(self.jet_ntuples, cut=f"({self.cuts}) & ({self.jet_cuts})", library='ak')
        
        data = ak.concatenate((tau_data, jet_data))
        return data
    
    def get_eff_rej(self):
        return pf.get_efficiency_and_rejection(self.y_true, self.y_pred)
        
    def get_tauid_wp_cut(self, wp_eff: int) -> str:
        correct_pred_taus = self.y_pred[self.y_true == 1]
        cut_val = np.percentile(correct_pred_taus, 100 - wp_eff)
        return f"{self.config.visualiser[self.type].y_pred} > {cut_val}"

    def get_tauid_rej_wp_cut(self, wp_eff: int) -> str:
        correct_pred_fakes = self.y_pred[self.y_true == 0]
        cut_val = np.percentile(correct_pred_fakes, wp_eff)
        return f"{self.config.visualiser[self.type].y_pred} > {cut_val}"
    
    def __getitem__(self, index: str) -> np.ndarray:
        return ak.to_numpy(self.data[index])
        
    
def visualise(config: DictConfig):
    
    # Make results dir
    results_dir = get_results_dir(config)
    run_dir = Path(results_dir).parents[0]
    plotting_dir = os.path.join(run_dir, "plots")
    os.makedirs(plotting_dir, exist_ok=True)
    
    # Make some visualisers
    tauid_utc_loader = ResultLoader(config, "UTC_TauID")
    tauid_rnn_loader = ResultLoader(config, "TauIDRNN")
    utc_loader = ResultLoader(config, "UTC")
    
    # Print some metrics
    labels = ['fakes', '1p0n', '1p1n', '1pxn', '3p0n', '3pxn']
    log.info("\n" + classification_report(np.argmax(utc_loader.y_true, axis=1), np.argmax(utc_loader.y_pred, axis=1), target_names=labels))

    # ROC Curves
    _, ax = pf.create_ROC_plot_template()
    ax.plot(*tauid_utc_loader.get_eff_rej(), label='UTC: all prong')
    ax.plot(*tauid_rnn_loader.get_eff_rej(), label='TauIDRNN all prong')
    ax.legend()
    plt.savefig(os.path.join(plotting_dir, 'ROC.png'), dpi=300)
    
    # Confusion Matrix
    pf.plot_confusion_matrix(utc_loader.y_true, utc_loader.y_pred, saveas=os.path.join(plotting_dir, "confusion_matrix.png"), title='UTC')
    pf.plot_confusion_matrix(utc_loader.y_true, utc_loader.y_pred_prev, saveas=os.path.join(plotting_dir, "confusion_matrix_current.png"), title='Current')
    labels = ["1p0n", "1p1n", "1pxn", "3p0n", "3pxn"]
    
    y_true = np.delete(utc_loader.y_true, 0, axis=1)
    y_pred = np.delete(utc_loader.y_pred_prev, 0, axis=1)
    pf.plot_confusion_matrix(y_true, y_pred, labels=labels, saveas=os.path.join(plotting_dir, "confusion_matrix_dmc.png"), title='DMC')
    
    utc_loader.change_cuts("TauJets_isRNNJetIDMedium == 1")
    y_true = np.delete(utc_loader.y_true, 0, axis=1)
    y_pred = np.delete(utc_loader.y_pred_prev, 0, axis=1)
    pf.plot_confusion_matrix(y_true, y_pred, labels=labels, saveas=os.path.join(plotting_dir, "confusion_matrix_dmc_tauID_medium.png"), title='DMC')
    
    y_true = np.delete(utc_loader.y_true, 0, axis=1)
    y_pred = np.delete(utc_loader.y_pred, 0, axis=1)
    pf.plot_confusion_matrix(y_true, y_pred, labels=labels, saveas=os.path.join(plotting_dir, "confusion_matrix_UTC_tauID_medium.png"), title='UTC')
    
    
    # Plot network outputs
    pf.plot_network_output(tauid_utc_loader, tauid_rnn_loader, os.path.join(plotting_dir, "NN_output.png"), title='network output')
    
    tauid_utc_loader.change_cuts("TauJets_nTracks == 1")
    tauid_rnn_loader.change_cuts("TauJets_nTracks == 1")
    pf.plot_network_output(tauid_utc_loader, tauid_rnn_loader, os.path.join(plotting_dir, "NN_output_1prong.png"), title='network output: 1-prong')
    
    tauid_utc_loader.change_cuts("TauJets_nTracks == 3")
    tauid_rnn_loader.change_cuts("TauJets_nTracks == 3")
    pf.plot_network_output(tauid_utc_loader, tauid_rnn_loader, os.path.join(plotting_dir, "NN_output_3prong.png"), title='network output: 3-prong')

    tauid_utc_loader.change_cuts(None)
    tauid_rnn_loader.change_cuts(None)
    
    # Get different ROC for each prongs
    _, ax = pf.create_ROC_plot_template("ROC UTC")
    ax.plot(*tauid_utc_loader.get_eff_rej(), label='UTC: all prong', color='b')
    tauid_utc_loader.change_cuts("TauJets_nTracks == 1")
    ax.plot(*tauid_utc_loader.get_eff_rej(), label='UTC: 1 prong', color='orange')
    tauid_utc_loader.change_cuts("TauJets_nTracks == 3")
    ax.plot(*tauid_utc_loader.get_eff_rej(), label='UTC: 3 prong', color='g')
    tauid_utc_loader.change_cuts(None)
    ax.legend()
    saveas = os.path.join(plotting_dir, 'ROC_utc_prongs.png')
    plt.savefig(saveas, dpi=300)
    log.info(f"Plotted {saveas}")
    
    _, ax = pf.create_ROC_plot_template("ROC TauIDRNN")
    ax.plot(*tauid_rnn_loader.get_eff_rej(), label='TauIDRNN: all prong', color='b')
    tauid_rnn_loader.change_cuts("TauJets_nTracks == 1")
    ax.plot(*tauid_rnn_loader.get_eff_rej(), label='TauIDRNN: 1 prong', color='orange')
    tauid_rnn_loader.change_cuts("TauJets_nTracks == 3")
    ax.plot(*tauid_rnn_loader.get_eff_rej(), label='TauIDRNN: 3 prong', color='g')
    ax.legend()
    tauid_rnn_loader.change_cuts(None)
    saveas = os.path.join(plotting_dir, 'ROC_rnn_prongs.png')
    plt.savefig(saveas, dpi=300)
    log.info(f"Plotted {saveas}")
    
    # Loop through each working point tauid efficiency 
    for wp in config.working_points:
        
        tauid_utc_cut = tauid_utc_loader.get_tauid_wp_cut(wp)
        tauid_rnn_cut = tauid_rnn_loader.get_tauid_wp_cut(wp)
        
        tauid_utc_loader.change_cuts(tauid_utc_cut)
        tauid_rnn_loader.change_cuts(tauid_rnn_cut)
        
        log.info(f"tauid_utc_cut @{wp} = {tauid_utc_cut}")
        log.info(f"tauid_rnn_cut @{wp} = {tauid_rnn_cut}")
        
        pf.plot_network_output(tauid_utc_loader, tauid_rnn_loader, os.path.join(plotting_dir, f"NN_output_eff_wp-{wp}.png"), title=f'network output @{wp}')
        
        # Get different ROC for each prongs
        _, ax = pf.create_ROC_plot_template("UTC_ROC_wps")
        ax.plot(*tauid_utc_loader.get_eff_rej(), label='UTC: all prong', color='b')
        tauid_utc_loader.change_cuts(f"(TauJets_nTracks == 1) & ({tauid_utc_cut})")
        ax.plot(*tauid_utc_loader.get_eff_rej(), label='UTC: 1 prong', color='orange')
        tauid_utc_loader.change_cuts(f"(TauJets_nTracks == 3) & ({tauid_utc_cut})")
        ax.plot(*tauid_utc_loader.get_eff_rej(), label='UTC: 3 prong', color='g')
        tauid_utc_loader.change_cuts(None)
        ax.legend(title=f'Efficiency = {wp}')
        saveas = os.path.join(plotting_dir, f'ROC_utc_prongs_{wp}-wp.png')
        plt.savefig(saveas, dpi=300)
        log.info(f"Plotted {saveas}")
        
        _, ax = pf.create_ROC_plot_template("TauIDRNN_ROC_wps")
        ax.plot(*tauid_rnn_loader.get_eff_rej(), label='TauIDRNN: all prong', color='b')
        tauid_rnn_loader.change_cuts(f"(TauJets_nTracks == 1) & ({tauid_rnn_cut})")
        ax.plot(*tauid_rnn_loader.get_eff_rej(), label='TauIDRNN: 1 prong', color='orange')
        tauid_rnn_loader.change_cuts(f"(TauJets_nTracks == 3) & ({tauid_rnn_cut})")
        ax.plot(*tauid_rnn_loader.get_eff_rej(), label='TauIDRNN: 3 prong', color='g')
        ax.legend(title=f'Efficiency = {wp}')
        tauid_rnn_loader.change_cuts(None)
        saveas = os.path.join(plotting_dir, f'ROC_rnn_prongs_{wp}-wp.png')
        plt.savefig(saveas, dpi=300)
        log.info(f"Plotted {saveas}")
        
        # Make confusion matrix for each WP cut
        # Plot confusion matrix with TauID cuts - drop the isFake column from y_true & y_pred
        labels = ["1p0n", "1p1n", "1pxn", "3p0n", "3pxn"]
        
        # With cut on TauID RNN
        utc_loader.change_cuts(tauid_rnn_cut)
        y_true = np.delete(utc_loader.y_true, 0, axis=1)
        y_pred = np.delete(utc_loader.y_pred, 0, axis=1)
        weights = utc_loader.weights
        utc_loader.change_cuts(None)
        
        saveas = os.path.join(plotting_dir, f"conf_matrix_TauIDRNN_{wp}.png")
        pf.plot_confusion_matrix(y_true, y_pred, saveas=saveas, labels=labels, title=f'Efficiency = {wp}')
        
        # With cut on UTC score
        utc_loader.change_cuts(tauid_utc_cut)
        y_true = np.delete(utc_loader.y_true, 0, axis=1)
        y_pred = np.delete(utc_loader.y_pred, 0, axis=1)
        weights = utc_loader.weights
        utc_loader.change_cuts(None)
        
        saveas = os.path.join(plotting_dir, f"conf_matrix_UTC_{wp}.png")
        pf.plot_confusion_matrix(y_true, y_pred, saveas=saveas, labels=labels, title=f'Efficiency = {wp}')
        
    # Now make efficiency plots for UTC
    for feature in config.sculpting_plots.keys():
        
        x_scale = config.sculpting_plots[feature].x_scale
        y_scale = config.sculpting_plots[feature].y_scale
        units =  config.sculpting_plots[feature].units
        binning = config.sculpting_plots[feature].bins
        
        _, ax = pf.create_plot_template(feature, y_label='efficiency', units=units, x_scale=x_scale, y_scale=y_scale,
                                        title=f'plots/{feature}_tauid_efficiencies.png')
        
        hist, bins = np.histogram(utc_loader[feature][utc_loader.y_true[:,0] == 0], bins=binning)
    
        # Loop through each working point tauid efficiency 
        for wp in config.working_points:
                
            tauid_utc_cut = tauid_utc_loader.get_tauid_wp_cut(wp)
            utc_loader.change_cuts(tauid_utc_cut)
            
            cut_hist, bins = np.histogram(utc_loader[feature][utc_loader.y_true[:,0] == 0], bins=bins)
                            
            ratio_hist = cut_hist / hist
            
            bincentres = [(bins[i]+bins[i+1])/2. for i in range(len(bins)-1)]
            ax.step(bincentres, ratio_hist , where='mid', label=f'WP = {wp}')

        ax.set_xlim((config.sculpting_plots[feature].min,  config.sculpting_plots[feature].max))
        
        utc_loader.change_cuts(None)
        ax.legend(title='UTC')
        saveas = os.path.join(plotting_dir, f'{feature}_tauid_efficiencies.png')
        plt.savefig(saveas, dpi=300)
        log.info(f"Plotted {saveas}")

    """
    ! Make sculpting plots for 1-prong
    """
        
    # Now make efficiency plots for UTC
    for feature in config.sculpting_plots.keys():
        
        utc_loader.change_cuts("TauJets_truthProng == 1")

        
        x_scale = config.sculpting_plots[feature].x_scale
        y_scale = config.sculpting_plots[feature].y_scale
        units =  config.sculpting_plots[feature].units
        binning = config.sculpting_plots[feature].bins
        
        _, ax = pf.create_plot_template(feature, y_label='efficiency', units=units, x_scale=x_scale, y_scale=y_scale,
                                        title=f'plots/{feature}_tauid_efficiencies.png')
        
        hist, bins = np.histogram(utc_loader[feature][utc_loader.y_true[:,0] == 0], bins=binning)
    
        # Loop through each working point tauid efficiency 
        for wp in config.working_points:
                
            tauid_utc_cut = tauid_utc_loader.get_tauid_wp_cut(wp)
            utc_loader.change_cuts(f"({tauid_utc_cut}) & (TauJets_truthProng == 1)")
            
            cut_hist, bins = np.histogram(utc_loader[feature][utc_loader.y_true[:,0] == 0], bins=bins)
                            
            ratio_hist = cut_hist / hist
            
            bincentres = [(bins[i]+bins[i+1])/2. for i in range(len(bins)-1)]
            ax.step(bincentres, ratio_hist , where='mid', label=f'WP = {wp}')

        ax.set_xlim((config.sculpting_plots[feature].min,  config.sculpting_plots[feature].max))
        
        utc_loader.change_cuts(None)
        ax.legend(title='UTC: 1-prong')
        saveas = os.path.join(plotting_dir, f'{feature}_tauid_efficiencies_1prong.png')
        plt.savefig(saveas, dpi=300)
        log.info(f"Plotted {saveas}")
        
    """
    ! Make sculpting plots for 3-prong
    """
        
    # Now make efficiency plots for UTC
    for feature in config.sculpting_plots.keys():
        
        utc_loader.change_cuts("TauJets_truthProng == 3")

        
        x_scale = config.sculpting_plots[feature].x_scale
        y_scale = config.sculpting_plots[feature].y_scale
        units =  config.sculpting_plots[feature].units
        binning = config.sculpting_plots[feature].bins
        
        _, ax = pf.create_plot_template(feature, y_label='efficiency', units=units, x_scale=x_scale, y_scale=y_scale,
                                        title=f'plots/{feature}_tauid_efficiencies.png')
        
        hist, bins = np.histogram(utc_loader[feature][utc_loader.y_true[:,0] == 0], bins=binning)
    
        # Loop through each working point tauid efficiency 
        for wp in config.working_points:
                
            tauid_utc_cut = tauid_utc_loader.get_tauid_wp_cut(wp)
            utc_loader.change_cuts(f"({tauid_utc_cut}) & (TauJets_truthProng == 3)")
            
            cut_hist, bins = np.histogram(utc_loader[feature][utc_loader.y_true[:,0] == 0], bins=bins)
                            
            ratio_hist = cut_hist / hist
            
            bincentres = [(bins[i]+bins[i+1])/2. for i in range(len(bins)-1)]
            ax.step(bincentres, ratio_hist , where='mid', label=f'WP = {wp}')

        ax.set_xlim((config.sculpting_plots[feature].min,  config.sculpting_plots[feature].max))
        
        utc_loader.change_cuts(None)
        ax.legend(title='UTC: 3-prong')
        saveas = os.path.join(plotting_dir, f'{feature}_tauid_efficiencies_3prong.png')
        plt.savefig(saveas, dpi=300)
        log.info(f"Plotted {saveas}")
        
        
        
        # Also make some control plots too
        _, ax = pf.create_plot_template(feature, units=units, x_scale=x_scale, y_scale=y_scale, title=f'plots/{feature}')
        ax.hist(utc_loader[feature], histtype='step', bins=100)
        saveas = os.path.join(plotting_dir, f'{feature}.png')
        plt.savefig(saveas, dpi=300)
        log.info(f"Plotted {saveas}")
    
    # Now make efficiency plots for TauID RNN
    for feature in config.sculpting_plots.keys():
        
        x_scale = config.sculpting_plots[feature].x_scale
        y_scale = config.sculpting_plots[feature].y_scale
        units =  config.sculpting_plots[feature].units
        binning = config.sculpting_plots[feature].bins
        
        _, ax = pf.create_plot_template(feature, y_label='efficiency', units=units, x_scale=x_scale, y_scale=y_scale,
                                        title=f'plots/{feature}_tauidrnn_efficiencies.png')
        
        hist, bins = np.histogram(utc_loader[feature][utc_loader.y_true[:,0] == 0], bins=binning)
    
        # Loop through each working point tauid efficiency 
        for wp in config.working_points:
            tauid_rnn_cut = tauid_rnn_loader.get_tauid_wp_cut(wp)
            utc_loader.change_cuts(tauid_rnn_cut)
            
            cut_hist, bins = np.histogram(utc_loader[feature][utc_loader.y_true[:,0] == 0], bins=bins)
            
            ratio_hist = cut_hist / hist
            
            bincentres = [(bins[i]+bins[i+1])/2. for i in range(len(bins)-1)]
            ax.step(bincentres, ratio_hist ,where='mid', label=f'WP = {wp}')
    
        utc_loader.change_cuts(None)
        ax.legend(title='TauIDRNN')
        saveas = os.path.join(plotting_dir, f'{feature}_tauidrnn_efficiencies.png')
        plt.savefig(saveas, dpi=300)
        log.info(f"Plotted {saveas}")

    
    """
    ! Make sculpting plots for 1-prong RNN
    """
        
    # Now make efficiency plots for UTC
    for feature in config.sculpting_plots.keys():
        
        utc_loader.change_cuts("TauJets_truthProng == 1")

        
        x_scale = config.sculpting_plots[feature].x_scale
        y_scale = config.sculpting_plots[feature].y_scale
        units =  config.sculpting_plots[feature].units
        binning = config.sculpting_plots[feature].bins
        
        _, ax = pf.create_plot_template(feature, y_label='efficiency', units=units, x_scale=x_scale, y_scale=y_scale,
                                        title=f'plots/{feature}_tauid_efficiencies.png')
        
        hist, bins = np.histogram(utc_loader[feature][utc_loader.y_true[:,0] == 0], bins=binning)
    
        # Loop through each working point tauid efficiency 
        for wp in config.working_points:
                
            tauid_utc_cut = tauid_rnn_loader.get_tauid_wp_cut(wp)
            utc_loader.change_cuts(f"({tauid_utc_cut}) & (TauJets_truthProng == 1)")
            
            cut_hist, bins = np.histogram(utc_loader[feature][utc_loader.y_true[:,0] == 0], bins=bins)
                            
            ratio_hist = cut_hist / hist
            
            bincentres = [(bins[i]+bins[i+1])/2. for i in range(len(bins)-1)]
            ax.step(bincentres, ratio_hist , where='mid', label=f'WP = {wp}')

        ax.set_xlim((config.sculpting_plots[feature].min,  config.sculpting_plots[feature].max))
        
        utc_loader.change_cuts(None)
        ax.legend(title='RNN: 1-prong')
        saveas = os.path.join(plotting_dir, f'{feature}_tauidRNN_efficiencies_1prong.png')
        plt.savefig(saveas, dpi=300)
        log.info(f"Plotted {saveas}")
        
    """
    ! Make sculpting plots for 3-prong
    """
        
    # Now make efficiency plots for UTC
    for feature in config.sculpting_plots.keys():
        
        utc_loader.change_cuts("TauJets_truthProng == 3")

        
        x_scale = config.sculpting_plots[feature].x_scale
        y_scale = config.sculpting_plots[feature].y_scale
        units =  config.sculpting_plots[feature].units
        binning = config.sculpting_plots[feature].bins
        
        _, ax = pf.create_plot_template(feature, y_label='efficiency', units=units, x_scale=x_scale, y_scale=y_scale,
                                        title=f'plots/{feature}_tauid_efficiencies.png')
        
        hist, bins = np.histogram(utc_loader[feature][utc_loader.y_true[:,0] == 0], bins=binning)
    
        # Loop through each working point tauid efficiency 
        for wp in config.working_points:
                
            tauid_utc_cut = tauid_rnn_loader.get_tauid_wp_cut(wp)
            utc_loader.change_cuts(f"({tauid_utc_cut}) & (TauJets_truthProng == 3)")
            
            cut_hist, bins = np.histogram(utc_loader[feature][utc_loader.y_true[:,0] == 0], bins=bins)
                            
            ratio_hist = cut_hist / hist
            
            bincentres = [(bins[i]+bins[i+1])/2. for i in range(len(bins)-1)]
            ax.step(bincentres, ratio_hist , where='mid', label=f'WP = {wp}')

        ax.set_xlim((config.sculpting_plots[feature].min,  config.sculpting_plots[feature].max))
        
        utc_loader.change_cuts(None)
        ax.legend(title='RNN: 3-prong')
        saveas = os.path.join(plotting_dir, f'{feature}_tauidRNN_efficiencies_3prong.png')
        plt.savefig(saveas, dpi=300)
        log.info(f"Plotted {saveas}")
        
        # Also make some control plots too
        _, ax = pf.create_plot_template(feature, units=units, x_scale=x_scale, y_scale=y_scale, title=f'plots/{feature}')
        ax.hist(utc_loader[feature], histtype='step', bins=100)
        saveas = os.path.join(plotting_dir, f'{feature}.png')
        plt.savefig(saveas, dpi=300)
        log.info(f"Plotted {saveas}")
    
    
    
    """
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    Rejection Plots
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    # Loop through each working point tauid efficiency 
    for wp in config.rejection_working_points:
        
        tauid_utc_cut = tauid_utc_loader.get_tauid_rej_wp_cut(wp)
        tauid_rnn_cut = tauid_rnn_loader.get_tauid_rej_wp_cut(wp)
        
        tauid_utc_loader.change_cuts(tauid_utc_cut)
        tauid_rnn_loader.change_cuts(tauid_rnn_cut)
        
        log.info(f"tauid_utc_rej_cut @{wp} = {tauid_utc_cut}")
        log.info(f"tauid_rnn_rej_cut @{wp} = {tauid_rnn_cut}")
        
        pf.plot_network_output(tauid_utc_loader, tauid_rnn_loader, os.path.join(plotting_dir, f"NN_output_rej_wp-{wp}.png"), title=f'network output @{wp}')
        
        # Get different ROC for each prongs
        _, ax = pf.create_ROC_plot_template("UTC_ROC_wps")
        ax.plot(*tauid_utc_loader.get_eff_rej(), label='UTC: all prong')
        tauid_utc_loader.change_cuts(f"(TauJets_nTracks == 1) & ({tauid_utc_cut})")
        ax.plot(*tauid_utc_loader.get_eff_rej(), label='UTC: 1 prong')
        tauid_utc_loader.change_cuts(f"(TauJets_nTracks == 3) & ({tauid_utc_cut})")
        ax.plot(*tauid_utc_loader.get_eff_rej(), label='UTC: 3 prong')
        tauid_utc_loader.change_cuts(None)
        ax.legend(title='UTC')
        saveas = os.path.join(plotting_dir, f'ROC_utc_prongs_rej_{wp}-wp.png')
        plt.savefig(saveas, dpi=300)
        log.info(f"Plotted {saveas}")
        
        _, ax = pf.create_ROC_plot_template("TauIDRNN_ROC_wps")
        ax.plot(*tauid_rnn_loader.get_eff_rej(), label='TauIDRNN: all prong')
        tauid_rnn_loader.change_cuts(f"(TauJets_nTracks == 1) & ({tauid_rnn_cut})")
        ax.plot(*tauid_rnn_loader.get_eff_rej(), label='TauIDRNN: 1 prong')
        tauid_rnn_loader.change_cuts(f"(TauJets_nTracks == 3) & ({tauid_rnn_cut})")
        ax.plot(*tauid_rnn_loader.get_eff_rej(), label='TauIDRNN: 3 prong')
        ax.legend(title='TauID RNN')
        tauid_rnn_loader.change_cuts(None)
        saveas = os.path.join(plotting_dir, f'ROC_rnn_prongs_rej_{wp}-wp.png')
        plt.savefig(saveas, dpi=300)
        log.info(f"Plotted {saveas}")
        
        # Make confusion matrix for each WP cut
        # Plot confusion matrix with TauID cuts - drop the isFake column from y_true & y_pred
        labels = ["1p0n", "1p1n", "1pxn", "3p0n", "3pxn"]
        
        # With cut on TauID RNN
        utc_loader.change_cuts(tauid_rnn_cut)
        y_true = np.delete(utc_loader.y_true, 0, axis=1)
        y_pred = np.delete(utc_loader.y_pred, 0, axis=1)
        weights = utc_loader.weights
        utc_loader.change_cuts(None)
        
        saveas = os.path.join(plotting_dir, f"conf_matrix_TauIDRNN_rej_{wp}.png")
        pf.plot_confusion_matrix(y_true, y_pred, saveas=saveas, labels=labels, title=f'TauIDRNN_rej_{wp}')
        
        # With cut on UTC score
        utc_loader.change_cuts(tauid_utc_cut)
        y_true = np.delete(utc_loader.y_true, 0, axis=1)
        y_pred = np.delete(utc_loader.y_pred, 0, axis=1)
        weights = utc_loader.weights
        utc_loader.change_cuts(None)
        
        saveas = os.path.join(plotting_dir, f"conf_matrix_UTC_rej_{wp}.png")
        pf.plot_confusion_matrix(y_true, y_pred, saveas=saveas, labels=labels, title=f'UTC_rej_{wp}')
    
    
    # Now make efficiency plots for UTC
    for feature in config.sculpting_plots.keys():
        
        x_scale = config.sculpting_plots[feature].x_scale
        y_scale = config.sculpting_plots[feature].y_scale
        units =  config.sculpting_plots[feature].units
        binning = config.sculpting_plots[feature].bins
        
        _, ax = pf.create_plot_template(feature, y_label='background rejection', units=units, x_scale=x_scale, y_scale=y_scale,
                                        title=f'plots/{feature}_tauid_rejection.png')
        
        utc_loader.change_cuts(None)
        tauid_utc_loader.change_cuts(None)
        tauid_rnn_loader.change_cuts(None)
        
        hist, bins = np.histogram(utc_loader[feature][utc_loader.y_true[:,0] == 1], bins=binning)
    
        # Loop through each working point tauid efficiency 
        for wp in config.rejection_working_points:
                
            tauid_utc_cut = tauid_utc_loader.get_tauid_rej_wp_cut(wp)
            utc_loader.change_cuts(tauid_utc_cut)
            
            cut_hist, bins = np.histogram(utc_loader[feature][utc_loader.y_true[:,0] == 1], bins=bins)
                            
            ratio_hist = 1 - cut_hist / hist
            
            bincentres = [(bins[i]+bins[i+1])/2. for i in range(len(bins)-1)]
            ax.step(bincentres, ratio_hist , where='mid', label=f'WP = {wp}')

        utc_loader.change_cuts(None)
        ax.legend(title='UTC')
        saveas = os.path.join(plotting_dir, f'{feature}_tauid_rejection.png')
        plt.savefig(saveas, dpi=300)
        log.info(f"Plotted {saveas}")
    
        # Also make some control plots too
        _, ax = pf.create_plot_template(feature, units=units, x_scale=x_scale, y_scale=y_scale, title=f'plots/{feature}')
        ax.hist(utc_loader[feature], histtype='step', bins=100)
        saveas = os.path.join(plotting_dir, f'{feature}.png')
        plt.savefig(saveas, dpi=300)
        log.info(f"Plotted {saveas}")
    
    # Now make efficiency plots for TauID RNN
    for feature in config.sculpting_plots.keys():
        
        x_scale = config.sculpting_plots[feature].x_scale
        y_scale = config.sculpting_plots[feature].y_scale
        units =  config.sculpting_plots[feature].units
        binning = config.sculpting_plots[feature].bins
        
        _, ax = pf.create_plot_template(feature, y_label='background rejection', units=units, x_scale=x_scale, y_scale=y_scale,
                                        title=f'plots/{feature}_tauidrnn_rejection.png')
        
        hist, bins = np.histogram(utc_loader[feature][utc_loader.y_true[:,0] == 0], bins=binning)
    
        # Loop through each working point tauid efficiency 
        for wp in config.working_points:
            tauid_rnn_cut = tauid_rnn_loader.get_tauid_rej_wp_cut(wp)
            utc_loader.change_cuts(tauid_rnn_cut)
            
            cut_hist, bins = np.histogram(utc_loader[feature][utc_loader.y_true[:,0] == 1], bins=bins)
            
            ratio_hist = 1 - cut_hist / hist
            
            bincentres = [(bins[i]+bins[i+1])/2. for i in range(len(bins)-1)]
            ax.step(bincentres, ratio_hist ,where='mid', label=f'WP = {wp}')
    
        utc_loader.change_cuts(None)
        ax.legend(title='TauIDRNN')
        saveas = os.path.join(plotting_dir, f'{feature}_tauidrnn_rejection.png')
        plt.savefig(saveas, dpi=300)
        log.info(f"Plotted {saveas}")
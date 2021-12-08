"""
Preprocessing
_________________________________________________________________________________________________
This is where the data pre-processing will go
"""

import os
import uproot
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
import numba as nb
import matplotlib.pyplot as plt
from tensorflow.keras.layers.experimental import preprocessing


class Reweighter:
    """
    This class computes the pT re-weighting coefficients by making histograms of TauJets.pt for both jets and taus. 
    The re-weighting coefficient is the ratio of the tau / jet histograms
    TODO: Make it so that the directories and cuts are not hardcoded
    """

    def __init__(self, ntuple_dir, prong=None):
        tau_files = glob.glob(os.path.join(ntuple_dir, "*Gammatautau*", "*.root"))
        jet_files = glob.glob(os.path.join(ntuple_dir, "*JZ*", "*.root"))

        assert len(tau_files) != 0 and len(jet_files) != 0, "The Reweighter found no files! Please check file path to NTuples"

        jet_cuts = "(TauJets_ptJetSeed > 15000.0) & (TauJets_ptJetSeed < 10000000.0)"
        tau_cuts = jet_cuts
        if prong is not None:
            tau_cuts = f"(TauJets_truthProng == {prong}) & " + jet_cuts 

        variable = "TauJets_ptJetSeed"
        tau_data = uproot.concatenate(tau_files, filter_name=variable, cut=tau_cuts, library='np')
        jet_data = uproot.concatenate(jet_files, filter_name=variable, cut=jet_cuts, library='np')

        jet_pt = jet_data[variable]
        tau_pt = tau_data[variable]

        # Binning
        xmax = max([np.amax(tau_pt), np.amax(jet_pt)])
        xmin = min([np.amin(tau_pt), np.amin(jet_pt)])
        bin_edges = np.linspace(xmin, xmax, 1000)

        """
        Need to use the bin centres as weights rather than the bin edges! Otherwise you get imperfect pT re-weighting!
        Use plt.hist rather than np.histogram for this
        """

        plt.ioff()  # Disable interactive plotting so we don't see anything
        fig = plt.figure()
        tau_hist, _, _ = plt.hist(tau_pt, bins=bin_edges)
        jet_hist, _, _ = plt.hist(jet_pt, bins=bin_edges)
        plt.close(fig)  # Close figure so that it doesn't interfere with future plots

        # Reweighting coefficient
        self.coeff = np.where(jet_hist > 0, tau_hist / (jet_hist + 1e-12), 1)
        self.bin_edges = bin_edges

    def reweight(self, jet_pt, strides=None):
        """
        Get an array of weights from an array of jet pTs. One weight is asigned per jet. For plotting re-weighted
        histograms of  nested data e.g. TauTracks.pT etc... use the strides option.
        :param jet_pt: Array of TauJets.pt
        :param strides: An array containing the multiplicity of the object for each jet. E.g. [4, 5, 10, 0 ...] for a
        set of jets where; the 1st jet has 4 tracks, the second 5, the third 10, the fourth 0 ... you get the idea
        (I Hope!). This way we can go from per jet to per object weights.
        :return: An array of weights
        """
        # Get an array of weights from an array of pTs
        if strides is None:
            return self.coeff[np.digitize(jet_pt, self.bin_edges)].astype(np.float32)
        else:
            arr_length = np.sum(strides)  # The total number of objects e.g. Tracks, PFO, etc...
            weights = self.coeff[np.digitize(jet_pt, self.bin_edges)].astype(np.float32)
            weights_ravelled = np.ones(arr_length) * -999  # <-- so that if something is wrong it is obvious
            start_pos = 0
            # Apply the same weighting for each object belonging to the same jet
            for i in range(0, len(strides)):
                weights_ravelled[start_pos: start_pos + strides[i]] = weights[i]
                start_pos = start_pos + strides[i]
            return weights_ravelled
"""
This is where the data pre-processing will go
"""

import uproot
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
from variables import variables_dictionary
import numba as nb
import ray
#ray.init()
import matplotlib.pyplot as plt

class Reweighter:

    def __init__(self, ntuple_dir):
        sig_files = glob.glob(f"{ntuple_dir}\\*Gammatautau*\\*.root")
        bkg_files = glob.glob(f"{ntuple_dir}\\*JZ*\\*.root")

        # print(len(bkg_files))
        # for file in bkg_files:
        #     print(file)

        bkg_cuts = "(TauJets.ptJetSeed > 15000.0) & (TauJets.ptJetSeed < 10000000.0) & (TauJets.ptRatioEflowApprox < 5) & (TauJets.etOverPtLeadTrk < 30)"
        sig_cuts = "(TauJets.truthProng == 1) & "+ bkg_cuts
        variable = "TauJets.ptJetSeed"
        sig_data = uproot.concatenate(sig_files, filter_name=variable, cut=sig_cuts, library='np')
        bkg_data = uproot.concatenate(bkg_files, filter_name=variable, cut=bkg_cuts, library='np')

        bkg_pt = bkg_data[variable]
        sig_pt = sig_data[variable]

        # bkg_pt = np.where(bkg_pt > 0, np.log10(bkg_pt), 0)
        # sig_pt = np.where(sig_pt > 0, np.log10(sig_pt), 0)

        # Binning
        #bin_edges = np.percentile(bkg_pt, np.linspace(0, 100, 50))
        xmax = max([np.amax(sig_pt), np.amax(bkg_pt)])
        xmin = min([np.amin(sig_pt), np.amin(bkg_pt)])
        bin_edges = np.linspace(xmin, xmax, 1000)

        # Reweighting coefficient
        # sig_hist, _ = np.histogram(sig_pt, bins=bin_edges, density=True)
        # bkg_hist, _ = np.histogram(bkg_pt, bins=bin_edges, density=True)

        """
        Need to use the bin centres as weights rather than the bin edges! Otherwise you get imperfect pT reweighting!
        Use plt.hist rather than np.histogram for this
        """

        plt.ioff()  # Disable interactive plotting so we don't see anything
        fig = plt.figure()
        sig_hist, _, _ = plt.hist(sig_pt, bins=bin_edges)
        bkg_hist, _, _ = plt.hist(bkg_pt, bins=bin_edges)
        plt.close(fig)  # Close figure so that it doesn't interfere with future plots

        self.coeff = np.where(bkg_hist > 0, sig_hist / (bkg_hist + 1e-12), 1)
        self.bin_edges = bin_edges

    def reweight(self, bkg_pt):
        return self.coeff[np.digitize(bkg_pt, self.bin_edges) - 1].astype(np.float32)

reweighter = Reweighter("E:\\NTuples\\TauClassifier")


# #@ray.remote
# @nb.njit()
# def calc_median(array):
# 	return np.median(array)
#
# @ray.remote
# def compute_stats(files, cuts, var):
#
# 	print(f"Computing median and IQR for {var} ... ")
#
# 	array = uproot.concatenate(files, cut=cuts, filter_name=var, library='np')[var]
# 	array = np.concatenate(array)
# 	tmp_stats_dict = {"median": None, "iqr": None}
# 	tmp_stats_dict["median"] = calc_median(array)
# 	q75, q25 = np.percentile(array.flatten(), [75, 25])
# 	tmp_stats_dict["iqr"] = q75 - q25
# 	print(tmp_stats_dict)
#
# 	return tmp_stats_dict
#
# class Rescaler:
#
# 	def __init__(self, ntuple_dir, variables_dictionary):
# 		files = glob.glob(f"{ntuple_dir}\\**\\*.root")
# 		self.stats_dict = {}
#
# 		cuts = "(TauJets.ptJetSeed > 15000.0) & (TauJets.ptJetSeed < 10000000.0)"
#
# 		for var_types in variables_dictionary:
# 			# for var in variables_dictionary[var_types]:
# 			# 	array = uproot.concatenate(files, cut=cuts, filter_name=var, library='np')[var]
# 			#
# 			# 	array = np.concatenate(array)
# 			#
# 			# 	tmp_stats_dict = {"median": None, "iqr": None}
# 			# 	tmp_stats_dict["median"] = calc_median(array)
# 			# 	q75, q25 = np.percentile(array.flatten(), [75, 25])
# 			# 	tmp_stats_dict["iqr"] = q75 - q25
# 			# 	self.stats_dict = {**self.stats_dict, **{var_types: tmp_stats_dict}}
# 			# 	print(tmp_stats_dict)
#
# 			futures = [compute_stats.remote(files, cuts, var) for var in variables_dictionary[var_types]]
# 			dicts = ray.get(futures)
#
# 			for d in dicts:
# 				self.stats_dict = {**self.stats_dict, **{var_types: d}}
#
# 	def rescale(self, arr, var):
# 		if self.stats_dict[var]["iqr"] != 0:
# 			return (arr - self.stats_dict[var]["median"]) / self.stats_dict[var]["iqr"]
# 		return arr
#
# recaler = Rescaler("E:\\NTuples\\TauClassifier", variables_dictionary)
"""
Efficiency vs variables
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import glob
import uproot
from config import cuts
from files import gammatautau_files, jz_files



def compute_efficiency_cut_values(y_pred, wp_efficiency):
	cut_val = np.percentile(y_pred, 100 - wp_efficiency)
	return cut_val


class EfficiencyRejectionPlot:

	def __init__(self, pred_files, root_files, variable, working_points, cut_values=None, n_test_points=10, cuts=None, log=False, rejection=False):

		self._variable = variable
		self._log = log
		self._working_points = working_points
		self._rejection = rejection

		# Load and concatenate predictions
		self.y_pred = []
		for file in pred_files:
			data = np.load(file)
			self.y_pred.append(data['arr_0'])
		self.y_pred = np.concatenate([arr for arr in self.y_pred])
		self.y_pred = 1 - self.y_pred[:, 0]  # Switch to probability event was a Tau

		# NN cut values
		if cut_values is not None:
			self._cut_values = cut_values

		else:
			# Compute NN output cut values based on efficiency working points
			self._cut_values = []
			assert working_points is not None
			for wp in working_points:
				cut_val = compute_efficiency_cut_values(self.y_pred, wp)
				self._cut_values.append(cut_val)

		# Load variable data
		arr = uproot.concatenate(root_files, filter_name=self._variable, library='np', cut=cuts)
		arr = arr[self._variable]

		# Make sure that arrays are the same lengths
		assert len(arr) == len(self.y_pred)

		# Compute the efficiency
		self.effs = []
		for cv in self._cut_values:
			arr_cut = arr[np.argwhere(self.y_pred > cv)]                       # Apply cut on NN output
			binning = np.linspace(np.amin(arr), np.amax(arr), n_test_points)   # Histogram binning
			self._test_points = binning
			if log:
				binning = np.logspace(np.log10(np.amin(arr)), np.log10(np.amax(arr)), num=n_test_points, base=10)

			# Efficiency is the ratio of the histograms
			arr_hist, _ = np.histogram(arr, binning)
			arr_cut_hist, _ = np.histogram(arr_cut, binning)

			self.effs.append(arr_cut_hist / arr_hist)

		self.effs = np.array(self.effs)

		# If we want to make a rejection plot then the rejection power if the inverse of efficiency
		if rejection:
			self.effs = 1 / (self.effs + 1e-12)

		print(self._cut_values)

	def plot(self, ncol=1, saveas=None, save=True):

		# Plots the efficiencies
		fig, ax = plt.subplots()
		for i in range(0, len(self._cut_values)):
			label = "{:.1f}% -- cut value = {:.6f}".format(self._working_points[i], self._cut_values[i])
			ax.scatter(self._test_points[1:], self.effs[i], label=label)

		fontP = FontProperties()
		fontP.set_size('medium')
		ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP, title="wp", ncol=ncol)
		plt.rcParams.update({'font.size': 14})
		ax.set_xlabel(self._variable)
		if self._rejection:
			ax.set_ylabel("rejection")
		else:
			ax.set_ylabel("efficiency")
		if self._log:
			ax.set_xscale('log')
		if self._rejection:
			ax.set_yscale('log')
		if save:
			if self._rejection:
				plt.savefig(f"plots\\{self._variable}_rejection.png", bbox_inches='tight')
			else:
				plt.savefig(f"plots\\{self._variable}_efficiency.png", bbox_inches='tight')
		elif saveas is not None:
			plt.savefig(saveas, bbox_inches='tight')
		plt.show()
		plt.close(fig)

	def __add__(self, plot):
		self.effs = np.concatenate((self.effs, plot.effs))




if __name__ == "__main__":

	gammatautau_pred_files = glob.glob(os.path.join("network_outputs","*24794883*"))
	gammatautau_root_files = gammatautau_files.file_list
	variable = "TauJets.ptJetSeed"
	working_points = [95, 85, 75, 60, 45]


	jz_pred_files = glob.glob(os.path.join("network_outputs", "*248*"))
	jz_pred_files.extend(glob.glob(os.path.join("network_outputs", "*247949*")))
	jz_root_files = jz_files.file_list

	EfficiencyRejectionPlot(gammatautau_pred_files, gammatautau_root_files, "TauJets.ptJetSeed", working_points, cuts=cuts['Gammatautau'], log=True, n_test_points=20).plot()
	EfficiencyRejectionPlot(gammatautau_pred_files, gammatautau_root_files, "TauJets.mu", working_points, cuts=cuts['Gammatautau'], n_test_points=20).plot()
	EfficiencyRejectionPlot(gammatautau_pred_files, gammatautau_root_files, "TauJets.etaJetSeed", working_points,
							cuts=cuts['Gammatautau'], n_test_points=20).plot()
	plot = EfficiencyRejectionPlot(gammatautau_pred_files, gammatautau_root_files, "TauJets.phiJetSeed", working_points,
							cuts=cuts['Gammatautau'], n_test_points=20)
	plot.plot()

	cut_values = plot.cut_values

	EfficiencyRejectionPlot(jz_pred_files, jz_root_files, "TauJets.ptJetSeed", working_points, cut_values=cut_values,
							cuts=cuts['JZ1'], log=True, n_test_points=20,rejection=True).plot()
	EfficiencyRejectionPlot(jz_pred_files, jz_root_files, "TauJets.mu", working_points, cut_values=cut_values,
							cuts=cuts['JZ1'], n_test_points=20, rejection=True).plot()
	EfficiencyRejectionPlot(jz_pred_files, jz_root_files, "TauJets.etaJetSeed", working_points, cut_values=cut_values,
							cuts=cuts['JZ1'], n_test_points=20, rejection=True).plot()
	EfficiencyRejectionPlot(jz_pred_files, jz_root_files, "TauJets.phiJetSeed", working_points, cut_values=cut_values,
							cuts=cuts['JZ1'], n_test_points=20, rejection=True).plot()
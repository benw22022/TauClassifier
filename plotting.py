"""
Plot histograms using a generator
"""

import matplotlib.pyplot as plt
import uproot

from DataGenerator import DataGenerator
from files import training_files
from variables import variables_dictionary
import config
from preprocessing import reweighter
import numpy as np
import glob
from utils import logger
import time
import ray
from plotting_config import limits_dict

@ray.remote
class histogram:

	def __init__(self, files, variable, xrange, bins=100, log=False, reweight=False, cuts=None, colour="black", label="", step_size=5e6):

		self.var = [variable]
		self.files = files
		self.cuts = cuts
		self.colour = colour
		self.label = label
		self.log = log
		self.reweight = reweight
		self.step_size = step_size

		if reweight:
			self.var = [variable, "TauJets.ptJetSeed"]
		self.histogram = np.zeros(bins, np.float32)
		self.binning = np.linspace(xrange[0], xrange[1], bins + 1)

	def make_hist(self):
		for batch in uproot.iterate(self.files, filter_name=self.var, step_size=int(self.step_size), cut=self.cuts, library='np'):
			arr = batch[self.var[0]]
			is_nested_arr = False
			try:
				arr = np.concatenate([array for array in arr]).flatten()
				is_nested_arr = True
			except ValueError:
				pass

			if self.log:
				arr = np.log10(arr)

			if self.reweight:

				if is_nested_arr:
					strides = [len(array) for array in batch[self.var[0]]]
					weights = reweighter.reweight(batch[self.var[1]], strides)
				else:
					weights = reweighter.reweight(batch[self.var[1]])
				hist, _ = np.histogram(arr, self.binning, weights=weights)
				self.histogram += hist

			else:
				hist, _ = np.histogram(arr, self.binning)
				self.histogram += hist
		return self

	def hist(self):
		return self.histogram

def plot_hists(histograms):

	fig, ax = plt.subplots()
	for hist in histograms:
		plt.hist(hist.binning[:-1], bins=hist.binning, weights=hist.histogram, label=hist.label, color=hist.colour, histtype="step")
	var = histograms[0].var[0]
	ax.set_xlabel(var)
	ax.legend()
	plt.savefig(f"plots\\{var}.png")
	plt.close(fig)

@ray.remote
def plot_dm_hist(var):
	start_time = time.time()
	tau_files = glob.glob("E:\\NTuples\\TauClassifier\\*Gammatautau*\\*.root")
	cuts_1p0n = "(TauJets.truthProng == 1) & (TauJets.truthDecayMode == 0) & (TauJets.ptJetSeed > 15000.0) & (TauJets.ptJetSeed < 10000000.0)"
	cuts_1p1n = "(TauJets.truthProng == 1) & (TauJets.truthDecayMode == 1) & (TauJets.ptJetSeed > 15000.0) & (TauJets.ptJetSeed < 10000000.0)"
	cuts_1pxn = "(TauJets.truthProng == 1) & (TauJets.truthDecayMode > 1) & (TauJets.ptJetSeed > 15000.0) & (TauJets.ptJetSeed < 10000000.0)"
	hist_1p0n = histogram.remote(tau_files, var, limits_dict[var], label="1p0n", cuts=cuts_1p0n, colour="orange")
	hist_1p1n = histogram.remote(tau_files, var, limits_dict[var], label="1p1n", cuts=cuts_1p1n, colour="red")
	hist_1pxn = histogram.remote(tau_files, var, limits_dict[var], label="1pxn", cuts=cuts_1pxn, colour="green")
	hists = [hist_1p0n, hist_1p1n, hist_1pxn]
	plot_hists(ray.get([hist.make_hist.remote() for hist in hists]))
	[ray.kill(hist) for hist in hists]
	del hists
	logger.log(f"Done: {var} Time taken: {time.time() - start_time}s")



if __name__ == "__main__":
	ray.init()

	tau_files = glob.glob("E:\\NTuples\\TauClassifier\\*Gammatautau*\\*.root")
	jet_files = glob.glob("E:\\NTuples\\TauClassifier\\*JZ*\\*.root")
	tau_cuts = "(TauJets.truthProng == 1) & (TauJets.ptJetSeed > 15000.0) & (TauJets.ptJetSeed < 10000000.0)"
	cuts_1p0n = "(TauJets.truthProng == 1) & (TauJets.truthDecayMode == 0) & (TauJets.ptJetSeed > 15000.0) & (TauJets.ptJetSeed < 10000000.0)"
	cuts_1p1n = "(TauJets.truthProng == 1) & (TauJets.truthDecayMode == 1) & (TauJets.ptJetSeed > 15000.0) & (TauJets.ptJetSeed < 10000000.0)"
	cuts_1pxn = "(TauJets.truthProng == 1) & (TauJets.truthDecayMode > 1) & (TauJets.ptJetSeed > 15000.0) & (TauJets.ptJetSeed < 10000000.0)"
	jet_cuts = "(TauJets.ptJetSeed > 15000.0) & (TauJets.ptJetSeed < 10000000.0)"

	# futures = ([plot_dm_hist.remote(var) for var in limits_dict])
	# results = ray.wait(futures, num_returns=5)

	for var in limits_dict:
		start_time = time.time()
		# jet_hist = histogram.remote(jet_files,  var, limits_dict[var], label="Jets", reweight=True, cuts=jet_cuts, colour="blue")
		# tau_hist = histogram.remote(tau_files,  var, limits_dict[var], label="Taus", cuts=tau_cuts, colour="orange")
		hist_1p0n = histogram.remote(tau_files, var, limits_dict[var], label="1p0n", cuts=cuts_1p0n, colour="orange")
		hist_1p1n = histogram.remote(tau_files, var, limits_dict[var], label="1p1n", cuts=cuts_1p1n, colour="red")
		hist_1pxn = histogram.remote(tau_files, var, limits_dict[var], label="1pxn", cuts=cuts_1pxn, colour="green")
		# hists = [jet_hist, hist_1p0n, hist_1p1n, hist_1pxn]
		hists = [hist_1p0n, hist_1p1n, hist_1pxn]
		# hists = [jet_hist, tau_hist]
		plot_hists(ray.get([hist.make_hist.remote() for hist in hists]))
		[ray.kill(hist) for hist in hists]
		del hists
		logger.log(f"Done: {var} Time taken: {time.time() - start_time}s")


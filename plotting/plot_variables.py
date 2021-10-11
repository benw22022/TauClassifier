"""
Plot variable distributions
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'                        # Sets Tensorflow Logging Level
import ray
ray.init()
from DataGenerator import DataGenerator
import matplotlib.pyplot as plt
import numpy as np
from files import training_files
from config import cuts
from variables import variables_dictionary, variables_list
import uproot
from variables import var_lims
import seaborn as sns



# @ray.remote
class HistIterator:

	def __init__(self, file_list, variables, cuts="", step_size=10000, nbins=50, label="Default"):

		# file = "E:\\NTuples\\TauClassifier\\user.bewilson.TauID.364702.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_v0_output.root\\user.bewilson.24794900._000001.output.root"

		# file = glob.glob("E:\\NTuples\\TauClassifier\\*.*\\*.root")

		self._iterator = uproot.iterate(file_list, filter_name=variables_list, library="np")

		# print(next(self._iterator))

		self._nbins = nbins
		self.hist_dict = dict.fromkeys(variables)
		self._variables = variables
		self.label = label
		self._var_max_min_dict = dict.fromkeys(variables)
		for variable in self._var_max_min_dict:
			self._var_max_min_dict[variable] = np.zeros(2, dtype='float32')
		self._step_size = step_size
		self._file_list = file_list

	def __next__(self):
		return next(self._iterator)

	def bin_data(self):
		for batch in self._iterator:
			for variable in self._variables:
				arr = batch[variable]
				try:
					arr = np.concatenate(arr)
				except ValueError:
					pass
				if self.hist_dict[variable] is None:
					self.hist_dict[variable] = np.histogram(arr)
				else:
					self.hist_dict[variable][0] = np.concatenate(self.hist_dict[variable][0], np.histogram(arr)[0])
					self.hist_dict[variable][1] = np.concatenate(self.hist_dict[variable][1], np.histogram(arr)[1])
		self.reset()

	def find_max_min(self):
		while True:
			try:
				batch = next(self._iterator)
			except StopIteration:
				break
			for variable in self._variables:
				arr = batch[variable]
				print(arr)
				min_val = np.amin(arr)
				max_val = np.amax(arr)
				if min_val < self._var_max_min_dict[variable][0]:
					self._var_max_min_dict[variable][0] = min_val
				if max_val < self._var_max_min_dict[variable][1]:
					self._var_max_min_dict[variable][1] = max_val

		self.reset()

	def reset(self):
		self._iterator = uproot.iterate(self._file_list, cut=cuts, filter_name=self._variables,
										step_size=self._step_size, library='np')

	def get_range(self):
		self.find_max_min()
		print(f"----- {self.label} ----- ")
		for variable in self._var_max_min_dict:
			min_val = self._var_max_min_dict[variable][0]
			max_val = self._var_max_min_dict[variable][1]
			print(f"{variable} --- range: {min_val} <-> {max_val}")
		return self._var_max_min_dict


if __name__ == "__main__":

	# Initialize Generators
	batch_generator = DataGenerator(training_files, variables_dictionary, nbatches=250, cuts=cuts)
	names = ["TauTracks", "NeutralPFO", "ShotPFO", "ConvTrack", "TauJets"]
	max_items = {"TauTracks": 8, "NeutralPFO": 3, "ShotPFO": 8, "ConvTrack": 4}
	for i in range(0, len(batch_generator)):
		batch, labels, weights = batch_generator[i]

		print(len(batch))

		# Loop over all variables types
		for j in range(0, len(batch)):
			# Loop over all variables in type
			for k in range(0, len(batch[j][:, ])):
				try:
					arr = batch[j][:, k]
				except IndexError:
					break

				jets_data = np.take(arr, np.argwhere(labels[:, 0] == 1))
				taus_data = np.take(arr, np.argwhere(labels[:, 0] == 0))
				jets_weight = np.take(weights, np.argwhere(labels[:, 0] == 1))
				taus_weight = np.take(weights, np.argwhere(labels[:, 0] == 0))

				if len(jets_data.shape) > 1:
					max_items = 20
					jets_weight_tmp = np.ones_like((jets_data.flatten()))
					for w in range(max_items, len(jets_weight)):
						jets_weight_tmp[w-max_items:w] = jets_weight[w]
					jets_weight = jets_weight_tmp

				if names[j] in ["ConvTrack", "ShotPFO", "NeutralPFO", "TauTracks"]:
					pass

				else:
					fig, ax = plt.subplots()
					njets, jets_bins, jets_patches = ax.hist(jets_data.flatten(), weights=jets_weight, bins=50, label="Jets", histtype="step", color="blue")
					ntaus, taus_bins, taus_patches = ax.hist(taus_data.flatten(), weights=taus_weight, bins=50, label="Taus", histtype="step", color="orange")

					# xmax = max([np.amax(taus_data), np.amax(jets_data)])
					# xmin = min([np.amin(taus_data), np.amin(jets_data)])
					# binning = np.linspace(xmin, xmax, 51)

					# tau_hist, _ = np.histogram(taus_data.flatten(), binning, weights=taus_weight)
					# jet_hist, _ = np.histogram(jets_data.flatten(), binning, weights=jets_weight)
					# ax.step(binning[:-1], tau_hist, label="taus", color='orange')
					# ax.step(binning[:-1], jet_hist, label="jets", color='blue')

					# njets, jets_bins, jets_patches = ax.hist(jets_data.flatten(), bins=50,
					# 										 label="Jets", histtype="step", color="blue")
					# ntaus, taus_bins, taus_patches = ax.hist(taus_data.flatten(), bins=50,
					# 										 label="Taus", histtype="step", color="orange")

					#weights=jets_weight
					#weights=taus_weight

					ax.set_xlabel(variables_dictionary[names[j]][k])
					ax.legend()
					if names[j] in ["ConvTrack", "ShotPFO", "NeutralPFO", "TauTracks"]:

						def min_val(arr):
							min_val = np.amin(np.where(arr.flatten() > -4, arr.flatten(), 0))
							return min_val

						def max_val(hist, bins):
							cutoff = 0
							for q in range(0, len(bins)):
								if bins[q] > -3:
									cutoff = q
									break
							return np.amax(hist[cutoff:])

						plt.gca().set_xlim(left=min([min_val(jets_data), min_val(taus_data)]))
						plt.gca().set_ylim(top=max([max_val(njets, jets_bins), max_val(ntaus, taus_bins)])*1.5)
					var = variables_dictionary[names[j]][k]
					plt.savefig(f"plots\\variables\\batch_1_{var}.png")
					plt.show()

"""
Plot variable distributions
"""

from DataGenerator import DataGenerator
import ray
import matplotlib.pyplot as plt
import numpy as np
from files import training_files
from variables import variables_dictionary
from config import cuts
from variables import variables_dictionary,variables_list
from preprocessing import pt_reweight
import uproot
import glob
import pickle
ray.init()


# @ray.remote
class HistIterator:

	def __init__(self, file_list, variables, cuts="", step_size=10000, nbins=50, label="Default"):

		#file = "E:\\NTuples\\TauClassifier\\user.bewilson.TauID.364702.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_v0_output.root\\user.bewilson.24794900._000001.output.root"

		#file = glob.glob("E:\\NTuples\\TauClassifier\\*.*\\*.root")

		self._iterator = uproot.iterate(file_list, filter_name=variables_list, library="np")

		#print(next(self._iterator))

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
		self._iterator = uproot.iterate(self._file_list, cut=cuts, filter_name=self._variables, step_size=self._step_size, library='np')

	def get_range(self):
		self.find_max_min()
		print(f"----- {self.label} ----- ")
		for variable in self._var_max_min_dict:
			min_val = self._var_max_min_dict[variable][0]
			max_val = self._var_max_min_dict[variable][1]
			print(f"{variable} --- range: {min_val} <-> {max_val}")
		return self._var_max_min_dict
# class MaxMinFinder:
#
# 	def __init__(self, file_handler_list, variables_list, cut=cuts):
# 		self._var_max_min_dict = dict.fromkeys(variables_list)
# 		self._variables = variables_list

if __name__ == "__main__":

	# Initialize Generators
	training_batch_generator = DataGenerator(training_files, variables_dictionary, nbatches=250, cuts=cuts,
											 label="Training Generator")
	names = ["TauTracks", "NeutralPFO", "ShotPFO", "ConvTrack", "TauJets"]
	for i in range(0, len(training_batch_generator)):
		batch, labels, _ = training_batch_generator[i]
		for j in range(0, len(batch)):
			for k in range(0, batch[j].shape[1]):

				arr = batch[j][:, k]

				print(np.amax(arr))

				print(arr.shape)
				print(np.argwhere(labels == 1).shape)

				jets_data = np.take(arr, np.argwhere(labels == 1))
				taus_data = np.take(arr, np.argwhere(labels == 0))

				fig, ax = plt.subplots()
				ax.hist(jets_data.flatten(), bins=500, label="Jets", histtype="step", color="blue")
				ax.hist(taus_data.flatten(), bins=500, label="Taus", histtype="step", color="orange")
				ax.set_xlabel(variables_dictionary[names[j]][k])
				ax.legend()
				plt.savefig(f"plots\\variables\\batch_1_{variables_dictionary[names[j]][k]}.png")
				plt.show()
		break


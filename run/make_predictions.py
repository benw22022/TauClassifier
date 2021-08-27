"""
For each NTuple make an array of predictions
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                     # Disables GPU
from DataLoader import DataLoader
import ray
import glob
from config import cuts, config_dict
from variables import variables_dictionary


def split_list(alist, wanted_parts=1):
	length = len(alist)
	return [alist[i * length // wanted_parts: (i + 1) * length // wanted_parts]
			for i in range(wanted_parts)]


if __name__ == "__main__":

	# Initialize Ray
	ray.init()

	# Load model
	model_config = config_dict
	model_config["shapes"]["TauTrack"] = (len(variables_dictionary["TauTracks"]),) + (10,)
	model_config["shapes"]["ConvTrack"] = (len(variables_dictionary["ConvTrack"]),) + (10,)
	model_config["shapes"]["NeutralPFO"] = (len(variables_dictionary["NeutralPFO"]),) + (10,)
	model_config["shapes"]["ShotPFO"] = (len(variables_dictionary["ShotPFO"]),) + (10,)
	model_config["shapes"]["TauJets"] = (len(variables_dictionary["TauJets"]),)

	model_weights = "data\\weights-06.h5"

	# Get GammaTauTau files
	files = glob.glob("E:\\NTuples\\TauClassifier\\*Gammatautau*\\*.root")

	# Make DataLoaders
	files = split_list(files, len(files)//5)   # split into groups of 5 to speed things up
	nbatches = 500

	for file_chunk in files:
		dataloaders = []
		for file in file_chunk:
			dl = DataLoader.remote(file, [file], 1, nbatches, variables_dictionary, cuts=cuts["Gammatautau"], no_gpu=True)
			dataloaders.append(dl)
		ray.get([dl.predict.remote(model_config, model_weights, save_predictions=True) for dl in dataloaders])
		for dl in dataloaders:
			ray.kill(dl)

	files = glob.glob("E:\\NTuples\\TauClassifier\\*JZ*\\*.root")
	files = split_list(files, len(files)//5)

	for file_chunk in files:
		dataloaders = []
		for file in file_chunk:
			dl = DataLoader.remote(file, [file], 1, nbatches, variables_dictionary, cuts=cuts["JZ1"], no_gpu=True)
			dataloaders.append(dl)
		ray.get([dl.predict.remote(model_config, model_weights, save_predictions=True) for dl in dataloaders])
		for dl in dataloaders:
			ray.kill(dl)

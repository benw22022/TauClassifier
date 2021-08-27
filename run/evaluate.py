"""
Evaluate.py
___________________________________________________________________
Compute predictions using a weights file
"""


import os
import ray
import glob
from config.config import cuts, config_dict
from config.variables import variables_dictionary
from config.files import gammatautau_files 
from scripts.DataLoader import DataLoader


def split_list(alist, wanted_parts=1):
    """
    Splits a list into list of smaller lists
    :param alist: A list to split up
    :param wanted_parts: Number of parts to split alist into
    :returns: A split up list
    """
	length = len(alist)
	return [alist[i * length // wanted_parts: (i + 1) * length // wanted_parts]
			for i in range(wanted_parts)]

def evaluate(weight_file, ncores=5)
    
    # Initialize Ray
	ray.init()

	# Load model
	model_config = config_dict
	model_config["shapes"]["TauTrack"] = (len(variables_dictionary["TauTracks"]),) + (10,)
	model_config["shapes"]["ConvTrack"] = (len(variables_dictionary["ConvTrack"]),) + (10,)
	model_config["shapes"]["NeutralPFO"] = (len(variables_dictionary["NeutralPFO"]),) + (10,)
	model_config["shapes"]["ShotPFO"] = (len(variables_dictionary["ShotPFO"]),) + (10,)
	model_config["shapes"]["TauJets"] = (len(variables_dictionary["TauJets"]),)

	model_weights = weight_file

	# Get GammaTauTau files
	files = gammatautau_files.file_list

	# Make DataLoaders
    assert ncores > 0, "Number of cores must be greater than zero"
    if ncores > len(files):
        ncores = len(files)

	files = split_list(files, len(files)//ncores)   # split into groups of 5 to speed things up
	nbatches = 500

	for file_chunk in files:
		dataloaders = []
		for file in file_chunk:
			dl = DataLoader.remote(file, [file], 1, nbatches, variables_dictionary, cuts=cuts["Gammatautau"], no_gpu=True)
			dataloaders.append(dl)
		ray.get([dl.predict.remote(model_config, model_weights, save_predictions=True) for dl in dataloaders])
		for dl in dataloaders:
			ray.kill(dl)

	files = jz_files.file_list
	files = split_list(files, len(files)//ncores)

	for file_chunk in files:
		dataloaders = []
		for file in file_chunk:
			dl = DataLoader.remote(file, [file], 1, nbatches, variables_dictionary, cuts=cuts["JZ1"], no_gpu=True)
			dataloaders.append(dl)
		ray.get([dl.predict.remote(model_config, model_weights, save_predictions=True) for dl in dataloaders])
		for dl in dataloaders:
			ray.kill(dl)




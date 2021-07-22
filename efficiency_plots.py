"""
Efficiency vs variables
"""
import matplotlib.pyplot as plt
from DataGenerator import DataGenerator
from keras.models import load_model
from sklearn.metrics import auc, roc_auc_score
from sklearn import metrics
import pandas as pd
import numpy as np
import seaborn as sns
import numba as nb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
import ray
import json
from files import testing_files
from variables import variables_dictionary
from models import ModelDSNN
from config import config_dict, cuts
import os
from tensorflow.keras.layers.experimental import preprocessing
from evaluateMK2 import get_efficiency_and_rejection


if __name__ == "__main__":
	ray.init()
	read = True
	plot = True
	jet_tau_comp = True
	dm_analy = True
	model_weights = "data\\weights-08.h5"

	y_pred = []
	y_true = []
	var_arr = []
	y_tauid = []
	weights = []

	testing_batch_generator = DataGenerator(testing_files, variables_dictionary, nbatches=50, cuts=cuts)
	model_config = config_dict
	model_config["shapes"]["TauTracks"] = (len(variables_dictionary["TauTracks"]),) + (10,)
	model_config["shapes"]["ConvTrack"] = (len(variables_dictionary["ConvTrack"]),) + (10,)
	model_config["shapes"]["NeutralPFO"] = (len(variables_dictionary["NeutralPFO"]),) + (10,)
	model_config["shapes"]["ShotPFO"] = (len(variables_dictionary["ShotPFO"]),) + (10,)
	model_config["shapes"]["TauJets"] = (len(variables_dictionary["TauJets"]),)
	#model = ModelDSNN(model_config)

	normalizers = {"TauTrack": preprocessing.Normalization(),
				   "NeutralPFO": preprocessing.Normalization(),
				   "ShotPFO": preprocessing.Normalization(),
				   "ConvTrack": preprocessing.Normalization(),
				   "TauJets": preprocessing.Normalization()}
	for batch in testing_batch_generator:
		normalizers["TauTrack"].adapt(batch[0][0])
		normalizers["NeutralPFO"].adapt(batch[0][1])
		normalizers["ShotPFO"].adapt(batch[0][2])
		normalizers["ConvTrack"].adapt(batch[0][3])
		normalizers["TauJets"].adapt(batch[0][4])
	testing_batch_generator.reset_generator()
	model = ModelDSNN(model_config, normalizers=normalizers)
	load_status = model.load_weights(model_weights, )


	variable = "TauJets.ptJetSeed"
	gamma_tautau_batch_generator = DataGenerator([testing_files[0]], variables_dictionary, nbatches=50, cuts=cuts, extra_return_var=variable)

	for i in range(0, len(gamma_tautau_batch_generator)):
			batch_tmp, y_true_tmp, weights_tmp, var_arr_tmp = gamma_tautau_batch_generator[i]
			y_pred_tmp = model.predict(batch_tmp)
			y_pred.append(y_pred_tmp)
			y_true.append(y_true_tmp)
			weights.append(weights_tmp)
			var_arr.append(var_arr_tmp)

	eff, rej = get_efficiency_and_rejection(y_true, y_pred)

	plt.plot(eff, var_arr)
	plt.show()


"""
Test
___________________________________________________________
Plot confusion matrix and ROC curve using testing dataset
"""

import os
import glob
import numpy as np
from config.files import testing_files, ntuple_dir
from config.variables import variables_dictionary
from config.config import config_dict, get_cuts, models_dict
from scripts.DataGenerator import DataGenerator
from scripts.preprocessing import Reweighter
from plotting.plotting_functions import get_efficiency_and_rejection, plot_ROC, make_confusion_matrix, plot_confusion_matrix

def test(args):
	"""
	Plots confusion matrix and ROC curve
	:param args: Args parsed by tauclassifier.py
	"""
	
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Sets Tensorflow Logging Level

	testing_files = testing_files.file_list

	y_pred = []
	y_true = []
	weights = []

	for test_file in testing_files:
		file_id = os.path.basename(test_file)
		y_pred_file = glob.glob(os.path.join("network_predictions", "predictions", f"*{file_id}*"))[0]
		y_true_file = glob.glob(os.path.join("network_predictions", "truth", f"*{file_id}*"))[0]
		weights_file = glob.glob(os.path.join("network_predictions", "weights", f"*{file_id}*"))[0]
		with np.load(y_pred_file, allow_pickle=True) as file:
			y_pred.append(file["arr_0"].astype("float32"))
		with np.load(y_true_file, allow_pickle=True) as file:
			y_true.append(file["arr_0"].astype("int"))
		with np.load(weights_file, allow_pickle=True) as file:
			weights.append(file["arr_0"])

	y_pred = np.concatenate([arr for arr in y_pred])
	y_true = np.concatenate([arr for arr in y_true])
	weights = np.concatenate([arr for arr in weights])

	# Plot confusion matrix
	plot_confusion_matrix(y_pred, y_true, prong=args.prong, weights=weights)

	# Plot ROC Curve
	true_jets = y_true[:, 0]
	pred_jets = y_pred[:, 0]
	plot_ROC(1 - true_jets, 1 - true_taus, weights=weights, title="ROC Curve: Tau-Jets",
			 saveas=os.path.join("plots","ROC_jets.png"))

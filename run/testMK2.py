"""
Test
___________________________________________________________
Plot confusion matrix and ROC curve using testing dataset
"""

import os
import glob
import numpy as np
from scripts.utils import logger
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

	y_pred = []
	y_true = []
	weights = []

	
    # Initialize objects
	reweighter = Reweighter(ntuple_dir, prong=args.prong)
	cuts = get_cuts(args.prong)

	testing_batch_generator = DataGenerator(testing_files, variables_dictionary, nbatches=50, cuts=cuts,
												reweighter=reweighter, prong=args.prong, label="Ranking Generator")

	testing_batch_generator.load_model(args.model, config_dict, args.weights)
	_, _, _, baseline_loss, baseline_acc = testing_batch_generator.predict(make_confusion_matrix=True, make_roc=True)

	logger.log(f"Testing Loss = {baseline_loss}		Testing Accuracy = {baseline_acc}")

	# # TODO Rework this to make it neater
	# files = []
	# for fh in testing_files:
	# 	for f in fh.file_list:
	# 		files.append(f)

	# for test_file in files:
	# 	file_id = os.path.basename(test_file)
	# 	y_pred_file = glob.glob(os.path.join("network_predictions", "predictions", f"*{file_id}*"))[0]
	# 	y_true_file = glob.glob(os.path.join("network_predictions", "truth", f"*{file_id}*"))[0]
	# 	weights_file = glob.glob(os.path.join("network_predictions", "weights", f"*{file_id}*"))[0]
	# 	with np.load(y_pred_file, allow_pickle=True) as file:
	# 		y_pred.append(file["arr_0"].astype("float32"))
	# 	with np.load(y_true_file, allow_pickle=True) as file:
	# 		y_true.append(file["arr_0"].astype("int"))
	# 	with np.load(weights_file, allow_pickle=True) as file:
	# 		weights.append(file["arr_0"])

	# y_pred = np.concatenate([arr for arr in y_pred])
	# y_true = np.concatenate([arr for arr in y_true])
	# weights = np.concatenate([arr for arr in weights])

	# # Plot confusion matrix
	# plot_confusion_matrix(y_pred, y_true, prong=args.prong, weights=weights)

	# # Plot ROC Curve
	# true_jets = y_true[:, 0]
	# pred_jets = y_pred[:, 0]
	# plot_ROC(1 - true_jets, 1 - true_jets, weights=weights, title="ROC Curve: Tau-Jets",
	# 		 saveas=os.path.join("plots","ROC_jets.png"))

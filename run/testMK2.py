import matplotlib.pyplot as plt
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
import os
from tensorflow.keras.layers.experimental import preprocessing

from config.files import testing_files, ntuple_dir, models
from config.variables import variables_dictionary
from model.models import ModelDSNN, SetTransformer
from config.config import config_dict, get_cuts
from scripts.DataGenerator import DataGenerator
from scripts.preprocessing import Reweighter


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disables GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Sets Tensorflow Logging Level



def get_efficiency_and_rejection(y_true, y_pred, weights):
	fpr_keras, tpr_keras, thresholds_keras = metrics.roc_curve(y_true, y_pred, sample_weight=weights)

	fpr_keras.sort()
	tpr_keras.sort()

	# Get AUC
	auc_keras = auc(fpr_keras, tpr_keras)
	print(f"AUC = {auc_keras}")

	nonzero = fpr_keras != 0  # Copies fpr array but removes all entries that are 0
	eff, rej = tpr_keras[nonzero], 1.0 / fpr_keras[nonzero]
	return eff, rej


def plot_ROC(y_true, y_pred, weights=None, title="ROC curve", saveas="ROC.svg"):

	eff, rej = get_efficiency_and_rejection(y_true, y_pred, weights)

	plt.figure(1)
	plt.plot([0, 1], [0, 1], 'k--')
	#plt.plot(fpr_keras, tpr_keras, label='Keras (AUC = {:.3f})'.format(auc_keras))
	plt.plot(eff, rej)#, label='AUC (area = {:.3f})'.format(auc_keras))
	# plt.xlabel('False positive rate')
	# plt.ylabel('True positive rate')
	plt.xlabel('Signal Efficiency')
	plt.ylabel('Background Rejection')
	plt.title(title)
	plt.ylim(1e0, 1e4)
	plt.legend(loc='best')
	plt.yscale("log")
	plt.savefig(saveas)
	plt.show()


@nb.njit()
def make_confusion_matrix(prediction, truth, weights=None):
	"""
	Function to make the confusion matrix
	Produces a 2D array which can look like this:
				 0      1      2	  3
	P   jets | .... | .... | .... | .... | 0
	R	1p0n | .... | .... | .... | .... | 1
	E	1p1n | .... | .... | .... | .... | 2
	D	1pxn | .... | .... | .... | .... | 3
		 	 | jets | 1p0n | 1p1n | 1pxn
				 T      R      U      E
	
	:param y_pred: Array of neural network predictions
	:param y_true: Correspondin array of truth data
	:param prong (optional, default=None): Number of prongs - determines the axis labels
	leave as None if you are classifiying 1 and 3 prongs together 
	:param weights (optional, default=None): An array of weights the same length and y_true and y_pred
	"""
	
    nclasses = prediction.shape[1]
	cm = np.zeros((nclasses, nclasses), dtype="float32")
    if weights is None:
        weights = np.ones_like(prediction)
	for pred, true, weight in zip(prediction, truth, weight):
		pred_max_idx = np.argmax(pred)
		truth_max_idx = np.argmax(true)        
		cm[pred_max_idx][truth_max_idx] += weights
	return cm


def plot_confusion_matrix(y_pred, y_true, prong=None, weights=None):
	"""
	Function to plot confusion matrix

	:param y_pred: Array of neural network predictions
	:param y_true: Correspondin array of truth data
	:param prong (optional, default=None): Number of prongs - determines the axis labels
	leave as None if you are classifiying 1 and 3 prongs together 
	:param weights (optional, default=None): An array of weights the same length and y_true and y_pred
	"""

	conf_matrix = make_confusion_matrix(y_pred, y_true)

	for i in range(0, y_pred.shape[1]):
		conf_matrix[:, i] = conf_matrix[:, i] / np.sum(y_true[:, i])

	fig = plt.figure()
	
	
	labels = ["jets", "1p0n", "1p1n", "1pxn", "3p0n", "3pxn"]
	if prong == 1:
		labels = ["jets", "1p0n", "1p1n", "1pxn"]
	if prong == 3:
		prong_3_labels = ["jets", "3p0n", "3pxn"]

	xticklabels = labels
	yticklabels = labels
	ax = sns.heatmap(conf_matrix, annot=True, cmap="Oranges", xticklabels=xticklabels, yticklabels=yticklabels,
					 fmt=".2")
	plt.xlabel("Truth")
	plt.ylabel("Prediction")
	plt.savefig(os.path.join("plots", "cm.png"))
	plt.show()
	plt.close(fig)


def test(args)

	# reweighter = Reweighter(ntuple_dir)
	# testing_batch_generator = DataGenerator(testing_files, variables_dictionary, nbatches=50, cuts=get_cuts(), reweighter=reweighter)

	# y_pred = []
	# y_true = []
	# weights = []

	# if read_from_cache:
	# 	model_config = config_dict
	# 	model = models[model](model_config)

	# 	for i in range(0, len(testing_batch_generator)):
	# 		batch_tmp, y_true_tmp, weights_tmp = testing_batch_generator[i]
	# 		y_pred_tmp = model.predict(batch_tmp)
	# 		y_pred.append(y_pred_tmp)
	# 		y_true.append(y_true_tmp)
	# 		weights.append(weights_tmp)
	# 	y_true = np.concatenate([arr for arr in y_true])
	# 	y_pred = np.concatenate([arr for arr in y_pred])
	# 	weights = np.concatenate([arr for arr in weights])
	# 	np.savez(os.path.join("cache", "y_pred.npz"), np.array(y_pred, dtype='object'))
	# 	np.savez(os.path.join("cache", "y_true.npz"), np.array(y_true, dtype='object'))
	# 	np.savez(os.path.join("cache", "weights.npz)", np.array(weights, dtype='object'))

	# else:
	# 	with np.load(os.path.join("cache", "y_pred.npz"), allow_pickle=True) as file:
	# 		y_pred = file["arr_0"].astype("float32")
	# 	with np.load(os.path.join("cache", "y_true.npz"), allow_pickle=True) as file:
	# 		y_true = file["arr_0"].astype("int")
	# 	with np.load(os.path.join("cache", "weights.npz"), allow_pickle=True) as file:
	# 		weights = file["arr_0"]

	# # Plot confusion matrix
	# plot_confusion_matrix(y_pred, y_true, prong=prong, weights=weights)

	# # Plot ROC Curve
	# true_jets = y_true[:, 0]
	# pred_jets = y_pred[:, 0]
	# plot_ROC(1 - true_jets, 1 - true_taus, weights=weights, title="ROC Curve: Tau-Jets",
	# 		 saveas=os.path.join("plots","ROC_jets.png"))

	
	testing_files = testing_files.file_list

	y_pred = []
	y_true = []
	weights = []

	for test_file in testing_files:
		file_id = os.path.basename(test_file)
		y_pred_file = glob.glob(os.path.join("network_predictions", "predictions", f"*file_id*"))[0]
		y_true_file = glob.glob(os.path.join("network_predictions", "truth", f"*file_id*"))[0]
		weights_file = glob.glob(os.path.join("network_predictions", "weights", f"*file_id*"))[0]
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
	plot_confusion_matrix(y_pred, y_true, prong=prong, weights=weights)

	# Plot ROC Curve
	true_jets = y_true[:, 0]
	pred_jets = y_pred[:, 0]
	plot_ROC(1 - true_jets, 1 - true_taus, weights=weights, title="ROC Curve: Tau-Jets",
			 saveas=os.path.join("plots","ROC_jets.png"))

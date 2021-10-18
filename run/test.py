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

from config.files import testing_files, ntuple_dir
from config.variables import variables_dictionary
from model.models import ModelDSNN, SetTransformer
from config.config import config_dict, get_cuts
from scripts.DataGenerator import DataGenerator
from scripts.preprocessing import Reweighter

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
def make_confusion_matrix(prediction, truth):
	"""
				 0      1      2	  3
	P   jets | .... | .... | .... | .... | 0
	R	1p0n | .... | .... | .... | .... | 1
	E	1p1n | .... | .... | .... | .... | 2
	D	1pxn | .... | .... | .... | .... | 3
		 	 | jets | 1p0n | 1p1n | 1pxn
				 T      R      U      E
	"""
	cm = np.zeros((6, 6), dtype="float32")
	for pred, true in zip(prediction, truth):
		# print(f"pred = {pred} -- true = {true}")
		pred_max_idx = np.argmax(pred)
		truth_max_idx = np.argmax(true)
		cm[pred_max_idx][truth_max_idx] += 1.0
	return cm


def plot_confusion_matrix(y_pred, y_true):

	conf_matrix = make_confusion_matrix(y_pred, y_true)

	print(conf_matrix)
	print(y_true.shape)

	for i in range(0, 6):
		conf_matrix[:, i] = conf_matrix[:, i] / np.sum(y_true[:, i])
	print(conf_matrix)

	fig = plt.figure()
	xticklabels = ["jets", "1p0n", "1p1n", "1pxn", "3p0n", "3pxn"]
	yticklabels = ["jets", "1p0n", "1p1n", "1pxn", "3p0n", "3pxn"]
	ax = sns.heatmap(conf_matrix, annot=True, cmap="Oranges", xticklabels=xticklabels, yticklabels=yticklabels,
					 fmt=".2")
	plt.xlabel("Truth")
	plt.ylabel("Prediction")
	plt.savefig("plots/cm.png")
	plt.show()
	plt.close(fig)

def test(network_weights, read_from_cache):
	
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disables GPU
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Sets Tensorflow Logging Level

	ray.init()
	read = True
	plot = True
	jet_tau_comp = True
	dm_analy = True
	model_weights = "network_weights/weights-13.h5"
	reweighter = Reweighter(ntuple_dir)
	testing_batch_generator = DataGenerator(testing_files, variables_dictionary, nbatches=50, cuts=get_cuts(), reweighter=reweighter)

	y_pred = []
	y_true = []
	weights = []

	if read:
		model_config = config_dict
		
		model = SetTransformer(model_config)
		# model = ModelDSNN(model_config)

		for i in range(0, len(testing_batch_generator)):
			batch_tmp, y_true_tmp, weights_tmp = testing_batch_generator[i]
			y_pred_tmp = model.predict(batch_tmp)
			y_pred.append(y_pred_tmp)
			y_true.append(y_true_tmp)
			weights.append(weights_tmp)
		y_true = np.concatenate([arr for arr in y_true])
		y_pred = np.concatenate([arr for arr in y_pred])
		weights = np.concatenate([arr for arr in weights])
		np.savez("cache/y_pred.npz", np.array(y_pred, dtype='object'))
		np.savez("cache/y_true.npz", np.array(y_true, dtype='object'))
		np.savez("cache/weights.npz", np.array(weights, dtype='object'))

	else:
		with np.load("cache/y_pred.npz", allow_pickle=True) as file:
			y_pred = file["arr_0"].astype("float32")
		with np.load("cache/y_true.npz", allow_pickle=True) as file:
			y_true = file["arr_0"].astype("int")
		with np.load("cache/weights.npz", allow_pickle=True) as file:
			weights = file["arr_0"]


	# Plot confusion matrix
	plot_confusion_matrix(y_pred, y_true)

	# Get truth arrays
	true_jets = y_true[:, 0]
	true_1p0n = y_true[:, 1]
	true_1p1n = y_true[:, 2]
	true_1pxn = y_true[:, 3]
	true_3p0n = y_true[:, 4]
	true_3pxn = y_true[:, 5]

	# Make array of the true taus
	true_tau_arrs = [true_1p1n, true_1pxn, true_3p0n, true_3pxn]
	true_taus = true_1p0n
	for arr in true_tau_arrs:
		true_taus = np.add(true_taus, arr)

	# Work out number of each class
	true_n_jets = np.sum(true_jets)
	true_n_1p0n = np.sum(true_1p0n)
	true_n_1p1n = np.sum(true_1p1n)
	true_n_1pxn = np.sum(true_1pxn)
	true_n_3p0n = np.sum(true_3p0n)
	true_n_3pxn = np.sum(true_3pxn)

	true_n_taus = np.sum(true_taus)

	print(f"Number of true jets = {true_n_jets}")
	print(f"Number of true taus = {true_n_taus}")
	print(f"--- Number of true 1p0n = {true_n_1p0n} ({true_n_1p0n / true_n_taus * 100} %)")
	print(f"--- Number of true 1p1n = {true_n_1p1n} ({true_n_1p1n / true_n_taus * 100} %)")
	print(f"--- Number of true 1pxn = {true_n_1pxn} ({true_n_1pxn / true_n_taus * 100} %)")
	print(f"--- Number of true 3p0n = {true_n_3p0n} ({true_n_3p0n / true_n_taus * 100} %)")
	print(f"--- Number of true 3pxn = {true_n_3pxn} ({true_n_3pxn / true_n_taus * 100} %)")

	#pred_jets = pred_taus = y_pred
	pred_jets = y_pred[:, 0]
	pred_1p0n = y_pred[:, 1]
	pred_1p1n = y_pred[:, 2]
	pred_1pxn = y_pred[:, 3]
	pred_3p0n = y_pred[:, 4]
	pred_3pxn = y_pred[:, 5]

	# true_1p0n_score = np.ma.masked_equal(y_pred[:, 1: 4] * true_1p0n, 0).compressed()
	# true_1p1n_score = np.ma.masked_equal(y_pred[:, 1: 4] * true_1p1n, 0).compressed()
	# true_1pxn_score = np.ma.masked_equal(y_pred[:, 1: 4] * true_1pxn, 0).compressed()
	# hv.extension('matplotlib')
	# (hv.Scatter3D(hv.Points(true_1p0n_score)) * hv.Scatter3D(hv.Points(true_1p1n_score)) * hv.Scatter3D(hv.Points(true_1pxn_score)))

	pred_taus = pred_1p0n + pred_1p1n + pred_1pxn + pred_3p0n +  pred_3pxn

	print(pred_taus)
	print(true_taus)

	true_pred_jets = []
	true_pred_taus = []
	for true_jet, pred in zip(true_jets, pred_jets):
		if true_jet == 1:
				true_pred_jets.append(pred)
		if true_jet == 0:
				true_pred_taus.append(pred)

	if plot:
		plot_ROC(true_taus, pred_taus, weights=weights, title="ROC Curve: Tau-Jets", saveas="plots/ROC_jets.png")
		plot_ROC(true_1p0n, pred_1p0n, title="ROC Curve: 1p0n", saveas="plots/ROC_1p0n.png")
		plot_ROC(true_1p1n, pred_1p1n, title="ROC Curve: 1p1n", saveas="plots/ROC_1p1n.png")
		plot_ROC(true_1pxn, pred_1pxn, title="ROC Curve: 1pxn", saveas="plots/ROC_1pxn.png")

		fig, ax = plt.subplots()
		ax.hist(np.array(true_pred_jets), range=(0, 1), histtype='step', label="jets prediction", color="blue")
#		ax.hist(np.array(true_pred_taus), range=(0, 1), histtype='step', label="taus prediction", color="orange")

		ax.hist(pred_1p0n, range=(0, 1), histtype='step', label="1p0n pred", color="orange")
		ax.hist(pred_1p1n, range=(0, 1), histtype='step', label="1p1n pred", color="red")
		ax.hist(pred_1pxn, range=(0, 1), histtype='step', label="1pxn pred", color="green")
		ax.hist(true_jets, range=(0, 1), histtype='step', label="jets true", color="blue", linestyle=('dashed'))
		ax.hist(true_1p0n, range=(0, 1), histtype='step', label="1p0n true", color="orange", linestyle=('dashed'))
		ax.hist(true_1p1n, range=(0, 1), histtype='step', label="1p1n true", color="red", linestyle=('dashed'))
		ax.hist(true_1pxn, range=(0, 1), histtype='step', label="1pxn true", color="green", linestyle=('dashed'))
		plt.legend()
		ax.set_xlabel("NN Output")
		plt.savefig("plots/response.png")
		plt.show()

	if jet_tau_comp:
		true_positive = 0       # Is a tau
		true_negative = 0        # Is a jet
		false_positive = 0
		false_negative = 0

		tau_score = []
		jet_score = []

		for jet_prediction, is_jet, is_tau in zip(pred_jets, true_jets, true_taus):
			if jet_prediction > 0.5: #tau_prediction:
				if is_jet == 1 and is_tau == 0:
					true_negative += 1
					jet_score.append(jet_prediction)
				elif is_jet == 0 and is_tau == 1:
					tau_score.append(jet_prediction)
					false_negative += 1
				elif is_jet == 1 and is_tau == 1:
					print(f"is_jet == {is_jet}  is_tau = {is_tau}")
				elif is_jet == 0 and is_tau == 0:
					print(f"is_jet == {is_jet}  is_tau = {is_tau}")
			if jet_prediction < 0.5:
				if is_jet == 0 and is_tau == 1:
					true_positive += 1
					tau_score.append(jet_prediction)
				elif is_jet == 1 and is_tau == 0:
					false_positive += 1
					jet_score.append(jet_prediction)
				elif is_jet == 1 and is_tau == 1:
					print(f"is_jet == {is_jet}  is_tau = {is_tau}")
				elif is_jet == 0 and is_tau == 0:
					print(f"is_jet == {is_jet}  is_tau = {is_tau}")

		tau_jet_precision = true_positive / (true_positive + false_positive)
		tau_jet_recall = true_positive / (true_positive + false_negative)

		tau_jet_f1_score = 2 * (tau_jet_precision * tau_jet_recall) / (tau_jet_precision + tau_jet_recall)

		print(f"Events tagged as jets = {true_positive + false_positive}")
		print(
			f"True number of jets = {true_n_jets} --- Correctly tagged jets = {true_negative} ({true_negative / true_n_jets * 100} %)"
			f" --- Incorrectly tagged jets = {false_negative} ({false_negative / true_n_jets * 100} %)")
		print(f"Events tagged as taus = {false_negative + true_negative}")
		print(
			f"True number of taus = {true_n_taus} --- Correctly tagged taus = {true_positive} ({true_positive / true_n_taus * 100} %) "
			f"--- Incorrectly tagged taus = {false_positive} ({false_positive / true_n_taus * 100} %)")
		print(f"Precision = {tau_jet_precision} --- Recall = {tau_jet_recall} --- F1 Score = {tau_jet_f1_score}")

		fig3, ax = plt.subplots()
		ax.hist(jet_score, range=(0, 1), histtype='step', label="jets", color="blue")
		ax.hist(tau_score, range=(0, 1), histtype='step', label="taus", color="orange")
		# ax.hist(true_jets, range=(0, 1), histtype='step', label="jets true", color="blue", linestyle=('dashed'))
		# ax.hist(true_taus, range=(0, 1), histtype='step', label="1-prong true", color="orange", linestyle=('dashed'))
		ax.set_xlabel("NN Response")
		ax.set_ylabel("Number of tau candidates")
		plt.yscale("log")
		plt.legend()
		plt.savefig("plots/jet_tau_response.png")
		plt.show()



	ray.shutdown()

if __name__ == "__main__":
	test()

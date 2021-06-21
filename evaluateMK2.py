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
ray.init()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disables GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Sets Tensorflow Logging Level


def plot_ROC(y_true, y_pred, weights=None, title="ROC curve", saveas="ROC.svg"):
	fpr_keras, tpr_keras, thresholds_keras = metrics.roc_curve(y_true, y_pred, sample_weight=weights)

	fpr_keras.sort()
	tpr_keras.sort()

	# Get AUC
	auc_keras = auc(fpr_keras, tpr_keras)
	print(f"AUC = {auc_keras}")
	plt.figure(1)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title(title)
	plt.legend(loc='best')
	plt.savefig(saveas)
	plt.show()


if __name__ == "__main__":

	read = True
	plot = True
	jet_tau_comp = True
	dm_analy = True
	model_weights = "data\\weights-05.h5"
	model_config = config_dict
	max_items = 20
	model_config["shapes"]["TauTracks"] = (len(variables_dictionary["TauTracks"]),) + (max_items,)
	model_config["shapes"]["ConvTrack"] = (len(variables_dictionary["ConvTrack"]),) + (max_items,)
	model_config["shapes"]["NeutralPFO"] = (len(variables_dictionary["NeutralPFO"]),) + (max_items,)
	model_config["shapes"]["ShotPFO"] = (len(variables_dictionary["ShotPFO"]),) + (max_items,)
	model_config["shapes"]["TauJets"] = (len(variables_dictionary["TauJets"]),)
	model = ModelDSNN(model_config)
	load_status = model.load_weights(model_weights)

	y_pred = []
	y_true = []
	weights = []

	if read:
		testing_batch_generator = DataGenerator(testing_files, variables_dictionary, nbatches=50, cuts=cuts)
		for i in range(0, len(testing_batch_generator)):
			batch_tmp, y_true_tmp, weights_tmp = testing_batch_generator[i]
			y_pred_tmp = model.predict(batch_tmp)
			y_pred.append(y_pred_tmp)
			y_true.append(y_true_tmp)
			weights.append(weights_tmp)
		np.savez("data\\y_pred.npz", np.array(y_pred, dtype='object'))
		np.savez("data\\y_true.npz", np.array(y_true, dtype='object'))
		np.savez("data\\weights.npz", np.array(weights, dtype='object'))

	else:
		with np.load("data\\y_pred.npz", allow_pickle=True) as file:
			y_pred = file["arr_0"]
		with np.load("data\\y_true.npz", allow_pickle=True) as file:
			y_true = file["arr_0"]
		with np.load("data\\weights.npz", allow_pickle=True) as file:
			weights = file["arr_0"]

	y_true = np.concatenate([arr for arr in y_true])
	y_pred = np.concatenate([arr for arr in y_pred])
	weights = np.concatenate([arr for arr in weights])

	#true_jets = true_taus = y_true
	true_jets = y_true[:, 0]
	true_1p0n = y_true[:, 1]
	true_1p1n = y_true[:, 2]
	true_1pxn = y_true[:, 3]
	true_taus = np.add(np.add(true_1p0n, true_1p1n), true_1pxn)

	true_n_jets = np.sum(true_jets)
	true_n_1p0n = np.sum(true_1p0n)
	true_n_1p1n = np.sum(true_1p1n)
	true_n_1pxn = np.sum(true_1pxn)
	true_n_taus = np.sum(true_taus)

	print(f"Number of true jets = {true_n_jets}")
	print(f"Number of true taus = {true_n_taus}")
	print(f"--- Number of true 1p0n = {true_n_1p0n} ({true_n_1p0n / true_n_taus * 100} %)")
	print(f"--- Number of true 1p1n = {true_n_1p1n} ({true_n_1p1n / true_n_taus * 100} %)")
	print(f"--- Number of true 1pxn = {true_n_1pxn} ({true_n_1pxn / true_n_taus * 100} %)")

	#pred_jets = pred_taus = y_pred
	pred_jets = y_pred[:, 0]
	pred_1p0n = y_pred[:, 1]
	pred_1p1n = y_pred[:, 2]
	pred_1pxn = y_pred[:, 3]
	pred_taus = np.add(pred_1p0n, pred_1p1n, pred_1pxn)

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
		plot_ROC(true_jets, pred_jets, title="ROC Curve: Jets", saveas="plots\\ROC_jets.png")
		plot_ROC(true_1p0n, pred_1p0n, title="ROC Curve: 1p0n", saveas="plots\\ROC_1p0n.png")
		plot_ROC(true_1p1n, pred_1p1n, title="ROC Curve: 1p1n", saveas="plots\\ROC_1p1n.png")
		plot_ROC(true_1pxn, pred_1pxn, title="ROC Curve: 1pxn", saveas="plots\\ROC_1pxn.png")

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
		plt.savefig("plots\\response.png")
		plt.show()

		# fig2, ax = plt.subplots()
		# ax.hist(pred_jets, range=(0, 1), histtype='step', label="jets", color="blue")
		# ax.hist(pred_taus, range=(0, 1), histtype='step', label="1-prong", color="orange")
		# ax.hist(true_jets, range=(0, 1), histtype='step', label="jets true", color="blue", linestyle=('dashed'))
		# ax.hist(true_taus, range=(0, 1), histtype='step', label="1-prong true", color="orange", linestyle=('dashed'))
		# plt.legend()
		# plt.savefig("plots\\jet_tau_response.png")
		# plt.show()

	if jet_tau_comp:
		true_positive = 0        # Is a jet
		true_negative = 0        # Is a tau
		false_positive = 0
		false_negative = 0

		tau_score = []
		jet_score = []

		for jet_prediction, is_jet, is_tau in zip(pred_jets, true_jets, true_taus):
			if jet_prediction > 0.5: #tau_prediction:
				if is_jet == 1 and is_tau == 0:
					true_positive += 1
					jet_score.append(jet_prediction)
				elif is_jet == 0 and is_tau == 1:
					tau_score.append(jet_prediction)
					false_positive += 1
				elif is_jet == 1 and is_tau == 1:
					print(f"is_jet == {is_jet}  is_tau = {is_tau}")
				elif is_jet == 0 and is_tau == 0:
					print(f"is_jet == {is_jet}  is_tau = {is_tau}")
			if jet_prediction < 0.5:
				if is_jet == 0 and is_tau == 1:
					true_negative += 1
					tau_score.append(jet_prediction)
				elif is_jet == 1 and is_tau == 0:
					false_negative += 1
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
			f"True number of jets = {true_n_jets} --- Correctly tagged jets = {true_positive} ({true_positive / true_n_jets * 100} %)"
			f" --- Incorrectly tagged jets = {false_negative} ({false_negative / true_n_jets * 100} %)")
		print(f"Events tagged as taus = {false_negative + true_negative}")
		print(
			f"True number of taus = {true_n_taus} --- Correctly tagged taus = {true_negative} ({true_negative / true_n_taus * 100} %) "
			f"--- Incorrectly tagged taus = {false_positive} ({false_positive / true_n_taus * 100} %)")
		print(f"Precision = {tau_jet_precision} --- Recall = {tau_jet_recall} --- F1 Score = {tau_jet_f1_score}")

		fig3, ax = plt.subplots()
		ax.hist(jet_score, range=(0, 1), histtype='step', label="jets", color="blue")
		ax.hist(tau_score, range=(0, 1), histtype='step', label="taus", color="orange")
		# ax.hist(true_jets, range=(0, 1), histtype='step', label="jets true", color="blue", linestyle=('dashed'))
		# ax.hist(true_taus, range=(0, 1), histtype='step', label="1-prong true", color="orange", linestyle=('dashed'))
		plt.legend()
		plt.savefig("plots\\jet_tau_response.png")
		plt.show()

	if dm_analy:
		cm = np.zeros((3, 3), dtype="float32")
		"""
					 0      1      2
		P	1p0n | .... | .... | .... 0
		R	1p1n | .... | .... | .... 1
		E	1pxn | .... | .... | .... 2
		D	 	 | 1p0n | 1p1n | 1pxn
					  T  R  U  E 				          
		"""

		for is_tau, is_1p0n, is_1p1n, is_1pxn, p_1p0n, p_1p1n, p_1pxn in zip(true_taus, true_1p0n, true_1p1n, true_1pxn,
																			 pred_1p0n, pred_1p1n, pred_1pxn):
			if is_tau == 1:
				if is_1p0n == 1:
					#print(f"TRUE 1p0n: pred_1p0n = {p_1p0n} --- pred_1p1n = {p_1p1n} --- pred_1pxn = {p_1pxn} ")
					if p_1p0n == max([p_1p0n, p_1p1n, p_1pxn]):
						cm[0][0] += 1
						print(cm)
					if p_1p1n == max([p_1p0n, p_1p1n, p_1pxn]):
						cm[0][1] += 1
					if p_1pxn == max([p_1p0n, p_1p1n, p_1pxn]):
						cm[0][2] += 1

				if is_1p1n == 1:
					#print(f"TRUE 1p1n: pred_1p0n = {p_1p0n} --- pred_1p1n = {p_1p1n} --- pred_1pxn = {p_1pxn} ")
					if p_1p0n == max([p_1p0n, p_1p1n, p_1pxn]):
						cm[1][0] += 1
					if p_1p1n == max([p_1p0n, p_1p1n, p_1pxn]):
						cm[1][1] += 1
					if p_1pxn == max([p_1p0n, p_1p1n, p_1pxn]):
						cm[1][2] += 1

				if is_1pxn == 1:
					#print(f"TRUE 1pxn: pred_1p0n = {p_1p0n} --- pred_1p1n = {p_1p1n} --- pred_1pxn = {p_1pxn} ")
					if p_1p0n == max([p_1p0n, p_1p1n, p_1pxn]):
						cm[2][0] += 1
					if p_1p1n == max([p_1p0n, p_1p1n, p_1pxn]):
						cm[2][1] += 1
					if p_1pxn == max([p_1p0n, p_1p1n, p_1pxn]):
						cm[2][2] += 1

		# Rescale for the actual numbers of 1p0n, 1p1n, 1pxn
		# cm[0, :] = cm[0, :] / true_n_1p0n
		# cm[1, :] = cm[1, :] / true_n_1p1n
		# cm[2, :] = cm[2, :] / true_n_1pxn

		print(cm)

"""
Evaluation of model goes here
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


def make_history_plots(history, metric, show_val=True, saveas=None):
	plt.plot(history.history[metric], label='train')
	if show_val:
		plt.plot(history.history[f"val_{metric}"], label='val')
	plt.xlabel('Epochs')
	plt.ylabel(metric)
	plt.legend()
	plt.show()
	if saveas is False:
		return
	elif saveas is None:
		plt.savefig(f"{metric}_history.svg")
	else:
		plt.savefig(saveas)


def m(x, w):
	"""Weighted Mean"""
	return np.sum(x * w) / np.sum(w)


def cov(x, y, w):
	"""Weighted Covariance"""
	return np.sum(w * (x - m(x, w)) * (y - m(y, w))) / np.sum(w)


def corr(x, y, w):
	"""Weighted Correlation"""
	return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))


def plot_heatmap(matrix_df, fnt_size, outfile):
	ax_dims = (40, 40)
	fig, ax = plt.subplots(figsize=ax_dims)

	sns_plot = sns.heatmap(matrix_df, annot=True, annot_kws={"fontsize": fnt_size}, cmap='coolwarm')
	sns.set(font_scale=3)

	# Get number of variables for title and outfile
	col_names = []
	for col in matrix_df.columns:
		col_names.append(col)
	ncols = len(col_names)

	print(col_names)
	print(ncols)

	title = outfile.replace(".png", "")
	title = title.replace("_", " ")
	title = title + ": " + str(ncols) + " variables"
	#outfile = str(ncols) + "_variabels_" + outfile

	# Thank you GitHub! https://github.com/mwaskom/seaborn/issues/1773
	# fix for mpl bug that cuts off top/bottom of seaborn viz
	b, t = plt.ylim()  # discover the values for bottom and top
	b += 0.5  # Add 0.5 to the bottom
	t -= 0.5  # Subtract 0.5 from the top
	plt.ylim(b, t)  # update the ylim(bottom, top) values
	# sns.set_context("paper", font_scale=0.01)
	ax.set_title(title)
	plt.show()  # ta-da!

	sns_plot.figure.savefig(outfile)


def plot_ROC(y_true, y_pred, weights):
	fpr_keras, tpr_keras, thresholds_keras = metrics.roc_curve(y_true, y_pred, sample_weight=weights)

	fpr_keras.sort()
	tpr_keras.sort()

	# Get AUC
	auc_keras = auc(fpr_keras, tpr_keras)
	plt.figure(1)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve')
	plt.legend(loc='best')
	plt.savefig("ROC.svg")
	plt.show()


#def make_predictions(testing_files_dictionary, variables_dictionary, model_weights, cuts=None):

@nb.njit()
def single_decision(y_arr):
	y_pred_v = []
	for pred in y_arr:
		max_val_idx = 0
		for i in range(0, 4):
			if pred[i] > max_val_idx:
				max_val_idx = i

		if max_val_idx == 0:
			y_pred_v.append("jets")
		if max_val_idx == 1:
			y_pred_v.append("1p0n")
		if max_val_idx == 2:
			y_pred_v.append("1p1n")
		if max_val_idx == 3:
			y_pred_v.append("1pxn")
	return y_pred_v


if __name__ == "__main__":
	import os
	# Disable GPU - cannot simultaneously use GPU for training and testing
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                     # Disables GPU
	from files import testing_files
	from variables import variables_dictionary
	from models import ModelDSNN
	from config import config_dict, cuts


	model_weights = "batch/training_2021-05-07_18.14.40/data/weights-05.h5"
	read = True

	model = ModelDSNN(config_dict)
	load_status = model.load_weights(model_weights)

	y_pred = []
	y_true = []
	weights = []

	if read:
		testing_batch_generator = DataGenerator(testing_files, variables_dictionary, nbatches=100, cuts=cuts)
		for i in range(0, len(testing_batch_generator)):
			batch_tmp, y_true_tmp, weights_tmp = testing_batch_generator[i]
			y_pred_tmp = model.predict(batch_tmp)
			y_pred.append(y_pred_tmp)
			y_true.append(y_true_tmp)
			weights.append(weights_tmp)
		np.savez("data/y_pred.npz", np.array(y_pred, dtype='object'))
		np.savez("data/y_true.npz", np.array(y_true, dtype='object'))
		np.savez("data/weights.npz", np.array(weights, dtype='object'))

	else:
		with np.load("data/y_pred.npz", allow_pickle=True) as file:
			y_pred = file["arr_0"]
		with np.load("data/y_true.npz", allow_pickle=True) as file:
			y_true = file["arr_0"]
		with np.load("data/weights.npz", allow_pickle=True) as file:
			weights = file["arr_0"]

	y_true = np.concatenate([arr for arr in y_true])
	y_pred = np.concatenate([arr for arr in y_pred])

	print(y_true.shape)
	print(y_pred.shape)
	nan_pos = np.argwhere(np.isnan(y_pred))
	nan_idx = []
	for pos in nan_pos:
		nan_idx.append(pos[0])

	@nb.njit()
	def count(y_arr):
		count_jets = 0
		count_1p0n = 0
		count_1p1n = 0
		count_1pxn = 0

		for y in y_arr:
			if y[0] == 1:
				count_jets += 1
			if y[1] == 1:
				count_1p0n += 1
			if y[2] == 1:
				count_1p1n += 1
			if y[3] == 1:
				count_1pxn += 1

		print(count_jets)
		print(count_1p0n)
		print(count_1p1n)
		print(count_1pxn)

	count(y_true)

	if len(nan_idx) > 0:
		y_pred_new = []
		y_true_new = []
		for i in range(0, len(y_true)):
			if i != nan_idx[0]:
				y_pred_new.append(y_pred[i])
				y_true_new.append(y_true[i])

		y_pred = np.array(y_pred_new)
		y_true = np.array(y_true_new)

		print(y_true.shape)
		print(y_pred.shape)



	results_dict = {"jets": y_true[:,0], "1p0n": y_true[:,1], "1p1n": y_true[:,2], "1pxn": y_true[:,3],
					"jets pred": y_pred[:,0], "1p0n pred": y_pred[:,1], "1p1n pred": y_pred[:,2], "1pxn pred": y_pred[:,3]}


	fig, ax = plt.subplots()
	ax.hist(results_dict["jets pred"], 5, histtype='step', label="jets", color="blue")
	ax.hist(results_dict["1p0n pred"], 5, histtype='step', label="1p0n", color="orange")
	ax.hist(results_dict["1p1n pred"], 5, histtype='step', label="1p1n", color="red")
	ax.hist(results_dict["1pxn pred"], 5, histtype='step', label="1pxn", color="green")
	plt.legend()
	plt.savefig("plots/response.svg")
	plt.show()




	y_pred_v = single_decision(y_pred)
	y_true_v = single_decision(y_true)

	confusion_mat = confusion_matrix(y_true_v, y_pred_v, labels=["jets",
																 "1p0n",
																 "1p1n",
																 "1pxn"])

	print(confusion_mat)
	confusion_mat[:, 0] = confusion_mat[:, 0] / len(results_dict["jets"])
	confusion_mat[:, 1] = confusion_mat[:, 1] / len(results_dict["1p0n"])
	confusion_mat[:, 2] = confusion_mat[:, 2] / len(results_dict["1p1n"])
	confusion_mat[:, 3] = confusion_mat[:, 3] / len(results_dict["1pxn"])
	print(confusion_mat)

	disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels = ["jets",
																					 "1p0n",
																					 "1p1n",
																					 "1pxn"])

	disp.plot()
	plt.savefig("plots/confusion_matrix.svg")
	plt.show()

	jet_to_jet = 0
	jet_to_p0n = 0
	jet_to_p1n = 0
	jet_to_pxn = 0
	
	p0n_to_jet = 0
	p0n_to_p0n = 0
	p0n_to_p1n = 0
	p0n_to_pxn = 0
	
	p1n_to_jet = 0
	p1n_to_p0n = 0
	p1n_to_p1n = 0
	p1n_to_pxn = 0
	
	pxn_to_jet = 0
	pxn_to_p0n = 0
	pxn_to_p1n = 0
	pxn_to_pxn = 0
	
	labels = ["jets", "1p0n", "1p1n", "1pxn"]
	labels2 = ["pred jets", "pred 1p0n", "pred 1p1n", "pred 1pxn"]
	
	cm = np.zeros((4,4))
	
	for truth, pred in zip(y_true_v, y_pred_v):
	
		if truth == "jets":
			if pred == "jets":
				jet_to_jet += 1
			if pred == "1p0n":
				jet_to_p0n += 1
			if pred == "1p1n":
				jet_to_p1n += 1
			if pred == "1pxn":
				jet_to_pxn += 1
	
		if truth == "1p0n":
			if pred == "jets":
				p0n_to_jet += 1
			if pred == "1p0n":
				p0n_to_p0n += 1
			if pred == "1p1n":
				p0n_to_p1n += 1
			if pred == "1pxn":
				p0n_to_pxn += 1
	
		if truth == "1p1n":
			if pred == "jets":
				p1n_to_jet += 1
			if pred == "1p0n":
				p1n_to_p0n += 1
			if pred == "1p1n":
				p1n_to_p1n += 1
			if pred == "1pxn":
				p1n_to_pxn += 1
	
		if truth == "1pxn":
			if pred == "jets":
				pxn_to_jet += 1
			if pred == "1p0n":
				pxn_to_p0n += 1
			if pred == "1p1n":
				pxn_to_p1n += 1
			if pred == "1pxn":
				pxn_to_pxn += 1
	
	
	cm[0][0] = jet_to_jet
	cm[0][1] = jet_to_p0n
	cm[0][2] = jet_to_p1n
	cm[0][3] = jet_to_pxn
	
	cm[1][0] = p0n_to_jet
	cm[1][1] = p0n_to_p0n
	cm[1][2] = p0n_to_p1n
	cm[1][3] = p0n_to_pxn
	
	cm[2][0] = p1n_to_jet
	cm[2][1] = p1n_to_p0n
	cm[2][2] = p1n_to_p1n
	cm[2][3] = p1n_to_pxn
	
	cm[3][0] = pxn_to_jet
	cm[3][1] = pxn_to_p0n
	cm[3][2] = pxn_to_p1n
	cm[3][3] = pxn_to_pxn
	
	print(cm)
	cm[:, 0] = cm[:, 0] / len(results_dict["jets"])
	cm[:, 1] = cm[:, 1] / len(results_dict["1p0n"])
	cm[:, 2] = cm[:, 2] / len(results_dict["1p1n"])
	cm[:, 3] = cm[:, 3] / len(results_dict["1pxn"])
	print(cm)
	
	matrix_df = pd.DataFrame(cm, columns=labels, index=labels2)
	
	plot_heatmap(matrix_df, 30, "plots/cm.svg")
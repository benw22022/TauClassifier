"""
Plotting Functions
________________________________________________________________________
File to store useful plotting functions
"""

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


def plot_ROC(y_true, y_pred, weights=None, title="ROC curve", saveas="ROC.png"):

	eff, rej = get_efficiency_and_rejection(y_true, y_pred, weights)

	plt.figure(1)
	plt.plot([0, 1], [0, 1], 'k--')
	#plt.plot(fpr_keras, tpr_keras, label='Keras (AUC = {:.3f})'.format(auc_keras))
	plt.plot(eff, rej)#, label='AUC (area = {:.3f})'.format(auc_keras))
	plt.xlabel('Signal Efficiency')
	plt.ylabel('Background Rejection')
	plt.ylim(1e0, 1e4)
	plt.yscale("log")
	plt.savefig(saveas)
	plt.title(title, loc='right', fontsize=5)


# @nb.njit()
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
		weights = np.ones(len(prediction), dtype='float32')

	for pred, true, weight in zip(prediction, truth, weights):
		pred_max_idx = np.argmax(pred)
		truth_max_idx = np.argmax(true)        
		cm[pred_max_idx][truth_max_idx] += weight
	return cm

def get_diag_score(conf_matrix, del_first=False):

	if del_first:
		np.delete(conf_matrix, 0, axis=0)
		np.delete(conf_matrix, 0, axis=1)

	return np.trace(conf_matrix) / conf_matrix.shape[0]
	
def plot_confusion_matrix(y_pred, y_true, prong=None, weights=None, saveas=None, title="", no_jets=False):
	"""
	Function to plot confusion matrix
	Makes two plots:
		One where columns are normalised to unity (Each cell corresponds to purity)
		One where rows are normalised to unity (Each cell corresponds to efficiency)
	
	:param y_pred: Array of neural network predictions
	:param y_true: Correspondin array of truth data
	:param prong (optional, default=None): Number of prongs - determines the axis labels
	leave as None if you are classifiying 1 and 3 prongs together 
	:param weights (optional, default=None): An array of weights the same length and y_true and y_pred
	:param saveas (optional, default=None): A filepath to save to
	:param title (optional, default=""): Title of the plot
	:param no_jets (optional, default=False): If True exclude jet catagory from plotting
	"""

	conf_matrix = make_confusion_matrix(y_pred, y_true, weights)

	# Normalise entries to unity
	purity_matrix = conf_matrix / conf_matrix.sum(axis=0, keepdims=1)        # nomalise columns to unity
	efficiency_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=1)    # nomalise rows to unity
	
	efficiency_matrix = np.nan_to_num(efficiency_matrix, posinf=0, neginf=0, copy=False).astype("float32")
	purity_matrix = np.nan_to_num(purity_matrix, posinf=0, neginf=0, copy=False).astype("float32")

	fig, (ax1, ax2) = plt.subplots(1,2, figsize=(40,15))

	labels = ["jets", "1p0n", "1p1n", "1pxn", "3p0n", "3pxn"]
	if prong == 1:
		labels = ["jets", "1p0n", "1p1n", "1pxn"]
	if prong == 3:
		labels = ["jets", "3p0n", "3pxn"]
	if no_jets:
		labels.remove("jets")

	xticklabels = labels
	yticklabels = labels
	sns.heatmap(efficiency_matrix, annot=True, cmap="Oranges", xticklabels=xticklabels, yticklabels=yticklabels,
						fmt=".2", vmin=0, vmax=1, ax=ax1, annot_kws={"size": 35 / np.sqrt(len(efficiency_matrix))},)
	sns.heatmap(purity_matrix, annot=True, cmap="Oranges", xticklabels=xticklabels, yticklabels=yticklabels,
						fmt=".2", vmin=0, vmax=1, ax=ax2, annot_kws={"size": 35 / np.sqrt(len(purity_matrix))},)
	sns.set(font_scale=8) 
	ax1.set_xlabel("Truth", fontsize=18)
	ax1.set_ylabel("Prediction", fontsize=18)
	ax2.set_xlabel("Truth", fontsize=18)
	ax2.set_ylabel("Prediction", fontsize=18)
	ax1.set_title(f"Diagonal Score = {get_diag_score(efficiency_matrix):.2f} Efficiency: {title}", loc='right', fontsize=12)
	ax2.set_title(f"Diagonal Score = {get_diag_score(purity_matrix):.2f} Purity: {title}", loc='right', fontsize=12)
	if saveas is None:
		plt.savefig(os.path.join("plots", "confusion_matrix.png"))
	else:
		plt.savefig(saveas)
	plt.clf()
	plt.close(fig)
	
	return np.trace(purity_matrix) / purity_matrix.shape[0]


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
	plt.ylim(1e0, 1e6)
	plt.yscale("log")
	plt.savefig(saveas)
	plt.title(title, loc='right', fontsize=5)
	plt.show()


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


def plot_confusion_matrix(y_pred, y_true, prong=None, weights=None, saveas=None, title="", no_jets=False):
	"""
	Function to plot confusion matrix

	:param y_pred: Array of neural network predictions
	:param y_true: Correspondin array of truth data
	:param prong (optional, default=None): Number of prongs - determines the axis labels
	leave as None if you are classifiying 1 and 3 prongs together 
	:param weights (optional, default=None): An array of weights the same length and y_true and y_pred
	"""

	conf_matrix = make_confusion_matrix(y_pred, y_true, weights)

	if weights is None:
		weights = np.ones_like(y_true)

	# Normalise entries to total amount of each class
	for i in range(0, y_pred.shape[1]):
		class_truth_total = np.sum(y_true[:, i])
		if class_truth_total == 0:
			class_truth_total = 1  # For if we have a dummy class still get nice plot
		conf_matrix[:, i] = conf_matrix[:, i] / class_truth_total

	fig = plt.figure()


	labels = ["jets", "1p0n", "1p1n", "1pxn", "3p0n", "3pxn"]
	if prong == 1:
		labels = ["jets", "1p0n", "1p1n", "1pxn"]
	if prong == 3:
		labels = ["jets", "3p0n", "3pxn"]
	if no_jets:
		labels.remove("jets")

	xticklabels = labels
	yticklabels = labels
	ax = sns.heatmap(conf_matrix, annot=True, cmap="Oranges", xticklabels=xticklabels, yticklabels=yticklabels,
						fmt=".2", vmin=0, vmax=1)
	plt.xlabel("Truth")
	plt.ylabel("Prediction")
	ax.set_title(title, loc='right', fontsize=5)
	if saveas is None:
		plt.savefig(os.path.join("plots", "confusion_matrix.png"))
	else:
		plt.savefig(saveas)
	plt.show()
	plt.close(fig)


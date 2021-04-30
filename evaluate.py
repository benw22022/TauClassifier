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
	outfile = str(ncols) + "_variabels_" + outfile

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


def make_predictions(testing_files_dictionary, variables_dictionary, model_weights, cuts=None):

	testing_batch_generator = DataGenerator(testing_files_dictionary, variables_dictionary, nbatches=1500, cuts=cuts)
	model = load_model(model_weights)

	y_pred = []
	y_true = []
	weights = []

	for i in range(0, len(testing_batch_generator)):
		batch_tmp, y_true_tmp = testing_batch_generator.load_batch(0)
		y_pred_tmp = model.predict(batch_tmp)
		y_pred.append(y_pred_tmp)
		y_true.append(y_true_tmp)
		weights.append(batch_tmp[2])

	plot_ROC(y_true, y_pred, weights)


	results_dict = {"jets": y_true[0], "1p0n": y_true[1], "1p1n": y_true[2], "1pXn": y_true[3],
					"jets pred": y_pred[0], "1p0n pred": y_pred[1], "1p1n pred": y_pred[2], "1pXn pred": y_pred[3]}

	# Output matrix
	weighted_corr = []

	# Loop over all variables and get the weighted correlation between them
	for i in range(0, len(y_pred)):
		x = y_true[i]
		w = weights[i]
		temp = []
		for j in range(0, len(y_true)):
			y = y_true[i]
			corr_xyw = corr(x, y, w)
			temp.append(float(corr_xyw))

		weighted_corr.append(temp)

	# Zip weighted correlations and variables together to make dictionary
	names_and_coorw = zip(results_dict.keys(), weighted_corr)
	corrw_dict = dict(names_and_coorw)

	# Convert dictionary to dataframe and appropriatly lable indexs and columns
	df_corrw = pd.DataFrame(corrw_dict, columns=["jets pred", "1p0n pred", "1p1n pred", "1pXn pred"],
										index=["jets", "1p0n", "1p1n", "1pXn"],)

	plot_heatmap(df_corrw, 4, "confusion_matrix.svg")


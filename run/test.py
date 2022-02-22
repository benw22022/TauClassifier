"""
Test
___________________________________________________________
Plot confusion matrix and ROC curve using testing dataset
"""
import os
import glob
from sklearn.model_selection import train_test_split
from source.utils import logger
from config.config import config_dict, models_dict
from config.variables import variable_handler
from config.files import testing_files
from run.train import build_dataset
import tensorflow as tf
import numpy as np
from plotting.plotting_functions import plot_confusion_matrix, plot_ROC, plot_1_and_3_prong_ROC
import ROOT
from source.DataLoader import DataLoader
from source.DataGenerator import DataGenerator
from source.preprocessing import Reweighter
from config.files import ntuple_dir
from config.config import get_cuts
import uproot
import tqdm
snapshotOptions = ROOT.RDF.RSnapshotOptions()
snapshotOptions.update = True
import sys

def evaluate_dataset(test_dataset, model, prong=None):
	
	if prong is not None:
		index = variable_handler.get_index("AUX", "TauJets.truthProng")
		test_dataset.unbatch().filter(lambda x, y, w, aux: aux[:, index] == prong).batch(50000)
	
	# Remove AUX data
	test_dataset = test_dataset.map(lambda x, y, weights, aux: (x, y, weights))

	y_pred = model.predict(test_dataset)
	y_true = np.concatenate([y for _, y, _ in test_dataset], axis=0)
	weights = np.concatenate([w for _, _, w in test_dataset], axis=0)

	return y_true, y_pred, weights

def test(args, roc_saveas=None, matrix_saveas=None, shuffle_index=None):
	"""
	Plots confusion matrix and ROC curve
	:param args: Args parsed by tauclassifier.py
	"""

	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

	model = models_dict[args.model](config_dict)
	opt = tf.keras.optimizers.Adam(learning_rate=1e-3)#args.lr) # default lr = 1e-3
	model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=[tf.keras.metrics.CategoricalAccuracy()])
	model.load_weights(args.weights)
        
	# test_dataset = build_dataset(os.path.join("data", "test_data", "*.dat"), batch_size=50000)
	# data_files = glob.glob("data/all_data/*.dat")
	# _, test_files = train_test_split(data_files, test_size=0.2, random_state=42)
	# test_dataset = build_dataset(test_files, aux_data=True, batch_size=50000)
	# test_dataset = build_dataset(glob.glob("data/test_data/*.dat"), aux_data=True, batch_size=50000)

	reweighter = Reweighter(ntuple_dir, prong=args.prong)
	cuts = get_cuts(args.prong)
	for fh in testing_files:
		print(fh)
		testing_batch_generator = DataGenerator([fh], variable_handler, batch_size=10000, cuts=cuts,
												reweighter=reweighter, prong=args.prong, label="Testing Generator")
		predictions = []
		for i in tqdm.tqdm(range(0, 2)):#len(testing_batch_generator))):
			p = model.predict(testing_batch_generator[i][0])
			predictions.append(p)
		
		print(predictions)
		predictions = np.concatenate([np.array(y) for y in predictions])	

		with uproot.update(fh.file_list[0]) as file:
			file["tree"] = {"TauClassifier_Score": predictions}
	
	sys.exit()
	# results = model.evaluate(test_dataset)
	# logger.log(f"test loss = {results[0]} , test acc  = {results[1]}")

	y_true, y_pred, weights = evaluate_dataset(test_dataset, model)
	y_true_1prong, y_pred_1prong, weights_1prong = evaluate_dataset(test_dataset, model, prong=1)
	y_true_3prong, y_pred_3prong, weights_3prong = evaluate_dataset(test_dataset, model, prong=3)

	# Baseline plots
	plot_ROC(y_true[:, 0], y_pred[: ,0], weights=weights, title=f"ROC: {os.path.basename(args.weights)}", saveas=roc_saveas)
	plot_1_and_3_prong_ROC((y_true_1prong[:, 0], y_pred_1prong[:, 0], weights_1prong), (y_true_3prong[:, 0], y_pred_3prong[:, 0], weights_3prong), title=f"ROC: {os.path.basename(args.weights)}") 
	plot_confusion_matrix(y_true, y_pred, weights=weights, title=f"Confusion Matrix: {os.path.basename(args.weights)}", saveas=matrix_saveas)
	

	return None #results

	"""
		if tauid_rnn_cut is not None:
		index = variable_handler.get_index("AUX", "isRNNJetIDSigTrans")
		test_dataset.filter(lambda x, y, weights, aux: aux[:, index] > tauid_rnn_cut)

		
	if sig_eff_cut is not None:
		cut_val = np.percentile(1 - y_pred[:, 0], sig_eff_cut)
		test_dataset = test_dataset.filter(lambda x, y, weights, aux: y[: ,0] < cut_val)
	"""

	# if shuffle_index is not None:

	# 	# @tf.function
	# 	# def shuffle(x, y, w):
	# 	# 	i, j = shuffle_index

	# 	# 	x_i_list = tf.unstack(x[i])
	# 	# 	x_ij_shuffled = tf.random.shuffle(x[i][j])
	# 	# 	x_i_tensors = []
	# 	# 	for idx, tensor in enumerate(x_i_list):
	# 	# 		if idx == j:
	# 	# 			tensor.append(x_ij_shuffled)
	# 	# 		else:
	# 	# 			x_i_tensors.append(tensor)

	# 	# 	x_i_stacked = tf.stack(x_i_tensors)

	# 	# 	new_x = []
	# 	# 	for idx, stacked_tensors in enumerate(x):
	# 	# 		if idx == i:
	# 	# 			new_x.append(x_i_stacked)
	# 	# 		else:
	# 	# 			new_x.append(stacked_tensors)

	# 	# 	return new_x, y, w

	# 	@tf.function
	# 	def shuffle(x, y, w):
	# 		a, b = shuffle_index
	# 		max_len = tf.shape(x[a])[1]
	# 		dtype = tf.float32
	# 		samples = tf.TensorArray(dtype=dtype, size=max_len, clear_after_read=False,  dynamic_size=True)
	# 		for i in tf.range(max_len):
	# 			if i == b:
	# 				samples.write(i, tf.random.shuffle(x[a][i]))
	# 			else:
	# 				samples = samples.write(i, x[a][i])
	# 		shuffled_samples = samples.stack()

	# 		new_x = []
	# 		for idx, stacked_tensors in enumerate(x):
	# 			if idx == i:
	# 				new_x.append(shuffled_samples)
	# 			else:
	# 				new_x.append(stacked_tensors)

	# 		return new_x, y, w

	# 	test_dataset.map(shuffle)

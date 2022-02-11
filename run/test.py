"""
Test
___________________________________________________________
Plot confusion matrix and ROC curve using testing dataset
"""

import enum
import os

from source.utils import logger
from config.config import config_dict, models_dict
from run.train import build_dataset
import tensorflow as tf
import numpy as np
from plotting.plotting_functions import plot_confusion_matrix, plot_ROC

def shuffle_index(ds, index):
	i, j = index

	ds_at_index = ds.map(lambda x, y, w: x[i][j]).shuffle(buffer_size=500)
	ds_at_index = ds.map(lambda x, y, w: x[i][j]).shuffle(buffer_size=500)






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
        
	test_dataset = build_dataset(os.path.join("data", "test_data", "*.dat"), batch_size=50000)

	if shuffle_index is not None:

		# @tf.function
		# def shuffle(x, y, w):
		# 	i, j = shuffle_index

		# 	x_i_list = tf.unstack(x[i])
		# 	x_ij_shuffled = tf.random.shuffle(x[i][j])
		# 	x_i_tensors = []
		# 	for idx, tensor in enumerate(x_i_list):
		# 		if idx == j:
		# 			tensor.append(x_ij_shuffled)
		# 		else:
		# 			x_i_tensors.append(tensor)

		# 	x_i_stacked = tf.stack(x_i_tensors)

		# 	new_x = []
		# 	for idx, stacked_tensors in enumerate(x):
		# 		if idx == i:
		# 			new_x.append(x_i_stacked)
		# 		else:
		# 			new_x.append(stacked_tensors)

		# 	return new_x, y, w

		@tf.function
		def shuffle(x, y, w):
			a, b = shuffle_index
			max_len = tf.shape(x[a])[1]
			dtype = tf.float32
			samples = tf.TensorArray(dtype=dtype, size=max_len, clear_after_read=False,  dynamic_size=True)
			for i in tf.range(max_len):
				if i == b:
					samples.write(i, tf.random.shuffle(x[a][i]))
				else:
					samples = samples.write(i, x[a][i])
			shuffled_samples = samples.stack()

			new_x = []
			for idx, stacked_tensors in enumerate(x):
				if idx == i:
					new_x.append(shuffled_samples)
				else:
					new_x.append(stacked_tensors)

			return new_x, y, w

		test_dataset.map(shuffle)


	results = model.evaluate(test_dataset)
	logger.log(f"test loss = {results[0]} , test acc  = {results[1]}")

	y_pred = model.predict(test_dataset)
	y_true = np.concatenate([y for _, y, _ in test_dataset], axis=0)
	weights = np.concatenate([w for _, _, w in test_dataset], axis=0)

	plot_ROC(y_true[:, 0], y_pred[: ,0], weights=weights, title=f"ROC: {os.path.basename(args.weights)}", saveas=roc_saveas)
	plot_confusion_matrix(y_true, y_pred, weights=weights, title=f"Confusion Matrix: {os.path.basename(args.weights)}", saveas=matrix_saveas)
	
	return results

	


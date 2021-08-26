"""
Benchmark loading times
"""
import os
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'    # Accelerated Linear Algebra (XLA) actually seems slower
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                     # Disables GPU
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'                       # Sets Tensorflow Logging Level
from variables import variables_dictionary
from DataGenerator import DataGenerator
from files import training_files
from utils import logger
from config import cuts
import numpy as np
import matplotlib.pyplot as plt
import ray
ray.init()
logger.set_log_level('INFO')


def benchmark():
	# Initialize Generators
	training_batch_generator = DataGenerator(training_files, variables_dictionary, nbatches=200, cuts=cuts,
											 label="Benchmark Generator", _benchmark=True)

	loading_times = []
	events_per_batch = []
	index_counter = []

	for i in range(0, len(training_batch_generator)):
		batch = training_batch_generator[i]
		events_per_batch.append(batch[2])
		loading_times.append(float(batch[3][5:]))
		index_counter.append(i)

	loading_times = np.array(loading_times, dtype="float32")
	events_per_batch = np.array(events_per_batch, dtype="float32")

	mean_loading_time = np.mean(loading_times)
	mean_events_per_batch = np.mean(events_per_batch)

	logger.log(f"Mean number of events per batch = {mean_events_per_batch}")
	logger.log(f"Mean loading time per batch = {mean_loading_time}")
	logger.log(f"Total time taken for {len(training_batch_generator)} batches = {np.sum(loading_times)}")

	fig, ax1 = plt.subplots()
	color = 'tab:orange'
	ax1.set_xlabel('Batch Number')
	ax1.set_ylabel('Batch Loading Time', color=color)
	ax1.plot(index_counter, loading_times, color=color)
	ax1.tick_params(axis='y', labelcolor=color)

	ax2 = ax1.twinx()

	color = 'tab:blue'
	ax2.set_ylabel('Events per batch', color=color)
	ax2.plot(index_counter, events_per_batch, color=color)
	ax2.tick_params(axis='y', labelcolor=color)

	fig.tight_layout()
	plt.show()
	# plt.savefig("plots\\M.2_Benchmark.svg")
	#
	# # Save data for comparison plots
	# np.save("data\\M.2_loading_times.npy", loading_times)
	# np.save("data\\M.2_events_per_batch.npy", events_per_batch)

if __name__ == "__main__":
	benchmark()
	ray.shutdown()

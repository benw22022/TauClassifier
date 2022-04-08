"""
Save data as tf.data.Dataset
____________________________________________________________________________________________
Saves batchs of data as tf.data.Datasets. Only way to avoid memory leaks with using uproot
in long training loops
"""

from source.datagenerator import DataGenerator
from source.build_features import make_feature_handler
from typing import List
import os
import tensorflow as tf
import tqdm
import yaml
import matplotlib
from multiprocessing import Process, Queue


def save(q, features_dataset, output_dir, i):
    q.put(tf.data.experimental.save(features_dataset, os.path.join(output_dir, f"data_{i:001d}.dat")))


def save_dataset(file_dict, output_dir: str, batch_size: int=100000):
    
    """
    Save Dataset
    _____________________________________________________________________________
    Uses DataGenerators (formally used for training) to generate batches of data
    which are converted to tf.data.Datasets which are then saved
    """
    
    file_dict = None
    with open("config/features_yaml", 'r') as stream:
        file_dict = yaml.load(stream, Loader=yaml.FullLoader)

    batch_generator = DataGenerator(file_dict, make_feature_handler, batch_size=50000)


    for i in tqdm.tqdm(range(0, len(batch_generator))):
    
        batch = batch_generator.load_batch(return_aux_vars=True)
        features_dataset = tf.data.Dataset.from_tensor_slices(batch)
    

        """
        In order to avoid a memory leak (Tensorflow **really** does not like you doing these operations in a loop)
        You have to spin up a seperate python process and run the function there - why? Haven't a clue!
        Oh and you have to have os.environ["CUDA_VISIBLE_DEVICES"] = "-1" else the dataset doesn't save properly!
        """
        queue = Queue()
        p = Process(target=save, args=(queue, features_dataset, output_dir, i))
        p.start()
        p.join()


if __name__ == "__main__":

    matplotlib.use('Agg')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    save_dataset(training_files, os.path.join("data", "train_data"))
    save_dataset(testing_files, os.path.join("data", "test_data"))
    save_dataset(validation_files, os.path.join("data", "val_data"))
    


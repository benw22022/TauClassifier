"""
Save Dataset
_____________________________________________________________________________
Uses DataGenerators (formally used for training) to generate batches of data
which are converted to tf.data.Datasets which are then saved
"""

import os
import tensorflow as tf
from config.variables import variable_handler
from source.DataGenerator import DataGenerator
from config.files import all_files, training_files, validation_files, testing_files, ntuple_dir
from config.config import get_cuts
from source.preprocessing import Reweighter
import tqdm
import matplotlib
from multiprocessing import Process, Queue

matplotlib.use('Agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def save(q, features_dataset, output_dir, i):        
    q.put(tf.data.experimental.save(features_dataset, os.path.join(output_dir, f"data_{i:001d}.dat")))


def save_dataset(file_handler_list, output_dir):
    """
    Generates large batches of data, converts them to a a tf.data.Dataset and then saves them to file
    Note: Uses tf.data.experimental.save - I would expect the syntax here to change eventually in the future
    args:
        file_handler_list: List[FileHandler] -> A list of FileHandlers 
        output_dir: str -> Filepath to output directory
    returns:
        None
    """

    reweighter = Reweighter(ntuple_dir, None)
    cuts = get_cuts(None)

    batch_generator = DataGenerator(file_handler_list, variable_handler, batch_size=50000, nbatches=50, cuts=cuts,
                                                reweighter=reweighter, prong=None, label="Training Generator")


    for i in tqdm.tqdm(range(0, len(batch_generator))):
    
        batch = batch_generator.load_batch(return_aux_vars=True)
        features_dataset = tf.data.Dataset.from_tensor_slices(batch)
        
        # tf.data.experimental.save(features_dataset, os.path.join(output_dir, f"data_{i:001d}.dat"))

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

    save_dataset(training_files, os.path.join("data", "train_data"))
    save_dataset(testing_files, os.path.join("data", "test_data"))
    save_dataset(validation_files, os.path.join("data", "val_data"))
    # save_dataset(all_files, os.path.join("data", "all_data"))
    
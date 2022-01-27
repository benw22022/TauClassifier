import os
# import ray
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
from config.variables import variable_handler
from scripts.DataGenerator import DataGenerator
from config.files import training_files, validation_files, ntuple_dir
from model.callbacks import ParallelModelCheckpoint
from scripts.utils import logger, get_number_of_events
from config.config import config_dict, get_cuts, models_dict
from scripts.preprocessing import Reweighter
import shutil
import uproot
import awkward as ak
import numba as nb
import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import matplotlib
matplotlib.use('Agg')
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' 

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array


def serialize_example(batch):
  """
  Creates a tf.train.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.train.Example-compatible
  # data type.
  feature = {
      'TauTracks': serialize_array(batch[0][0]),
      'NeutralPFO': serialize_array(batch[0][1]),
      'ShotPFO': serialize_array(batch[0][2]),
      'ConvTrack': serialize_array(batch[0][3]),
      'TauJets': serialize_array(batch[0][4]),
      'Label': _int64_feature(batch[1]),
      'Weight': _float_feature(batch[2]),
  }

  # Create a Features message using tf.train.Example.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

    
if __name__ == "__main__":

    reweighter = Reweighter(ntuple_dir, None)

    cuts = get_cuts(None)

    training_batch_generator = DataGenerator(validation_files, variable_handler, batch_size=50000, nbatches=50, cuts=cuts,
                                                reweighter=reweighter, prong=None, label="Training Generator")


    for i in tqdm.tqdm(range(0, len(training_batch_generator))):
    
        batch = training_batch_generator[i]
        features_dataset = tf.data.Dataset.from_tensor_slices(batch)
        
        tf.data.experimental.save(features_dataset, os.path.join("data", "val_data", f"data_{i:001d}.dat"))

        # print(f"Written {i} / {len(training_batch_generator)} files")


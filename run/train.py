"""
Main Code Body
"""
import os

# Do these things first before importing
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'    # Accelerated Linear Algebra (XLA) actually seems slower
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                     # Disables GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'                # Allow tensorflow to use more GPU VRAM

import ray
import json
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers.experimental import preprocessing
import pickle
import sys

from config.variables import variables_dictionary
from model.models import ModelDSNN, SetTransformer
from scripts.DataGenerator import DataGenerator
from config.files import training_files, validation_files, ntuple_dir
from model.callbacks import ParallelModelCheckpoint
from scripts.utils import logger
from config.config import config_dict, get_cuts, models
from scripts.preprocessing import Reweighter


def train(prong=None, log_level='INFO', model="DSNN", tf_log_level='2'):

    # Set log levels
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = tf_log_level # Sets Tensorflow Logging Level
    logger.set_log_level(log_level)

    # Initialize ray
    ray.init(include_dashboard=True)
    

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Initialize Generators
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    reweighter = Reweighter(ntuple_dir, prong=prong)

    cuts = get_cuts(prong)

    training_batch_generator = DataGenerator(training_files, variables_dictionary, nbatches=250, cuts=cuts,
                                             reweighter=reweighter, prong=prong, label="Training Generator")

    validation_batch_generator = DataGenerator(validation_files, variables_dictionary, nbatches=50, cuts=cuts,
                                               reweighter=reweighter, prong=prong, label="Validation Generator")

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Initialize Model
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    # Configure model
    model_config = config_dict
    model_config["shapes"]["TauTrack"] = (len(variables_dictionary["TauTracks"]),) + (10,)
    model_config["shapes"]["ConvTrack"] = (len(variables_dictionary["ConvTrack"]),) + (10,)
    model_config["shapes"]["NeutralPFO"] = (len(variables_dictionary["NeutralPFO"]),) + (10,)
    model_config["shapes"]["ShotPFO"] = (len(variables_dictionary["ShotPFO"]),) + (10,)
    model_config["shapes"]["TauJets"] = (len(variables_dictionary["TauJets"]),)

    model = models[model](model_config)

    # Configure callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss", min_delta=0.0001,
        patience=10, verbose=0, restore_best_weights=True)

    model_checkpoint = ParallelModelCheckpoint(model, path=os.path.join("network_weights", 'weights-{epoch:02d}.h5'),
                                               monitor="val_loss", save_best_only=False, save_weights_only=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=3, min_lr=4e-6)

    callbacks = [early_stopping, model_checkpoint, reduce_lr]

    # Compile and summarise model
    model.summary()

    njets = 9647968
    n1p0n = 1962842
    n1p1n = 4773455
    n1pxn = 2092967
    n3p0n = 1913289
    n3pxn = 1055423
    total = njets + n1p0n + n1p1n + n1pxn + n3p0n + n3pxn

    weight_for_jets = (1 / njets) * (total / 2.0)
    weight_for_1p0n = (1 / n1p0n) * (total / 2.0)
    weight_for_1p1n = (1 / n1p1n) * (total / 2.0)
    weight_for_1pxn = (1 / n1pxn) * (total / 2.0)
    weight_for_3p0n = (1 / n3p0n) * (total / 2.0)
    weight_for_3p1n = (1 / n3pxn) * (total / 2.0)

    class_weight = {0: weight_for_jets,
                    1: weight_for_1p0n,
                    2: weight_for_1p1n,
                    3: weight_for_1pxn,
                    4: weight_for_3p0n,
                    5: weight_for_3p1n,
                    }
    model.layers[-1].bias.assign([np.log(njets/(total)),
                                  np.log(n1p0n/(total)),
                                  np.log(n1p1n/(total)),
                                  np.log(n1pxn/(total)),
                                  np.log(n3p0n / (total)),
                                  np.log(n3pxn / (total)),
                                  ])


    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=[tf.keras.metrics.CategoricalAccuracy()])

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
     Train Model
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    history = model.fit(training_batch_generator, epochs=100, callbacks=callbacks, class_weight=class_weight,
                        validation_data=validation_batch_generator, validation_freq=1, verbose=1, shuffle=True,
                        steps_per_epoch=len(training_batch_generator))

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Make Plots 
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    # Loss History
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join("plots", "loss_history.svg"))
    plt.show()

    # Accuracy history
    plt.plot(history.history['categorical_accuracy'], label='train')
    plt.plot(history.history['val_categorical_accuracy'], label='val')
    plt.xlabel('Epochs')
    plt.ylabel('Categorical Accuracy')
    plt.legend()
    plt.savefig(os.path.join("plots", "accuracy_history.svg"))
    plt.show()

    return 0


if __name__ == "__main__":
    train()

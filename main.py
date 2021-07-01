"""
Main Code Body
"""
import os

# Do these things first before importing
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'    # Accelerated Linear Algebra (XLA) actually seems slower
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                     # Disables GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'                        # Sets Tensorflow Logging Level
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import ray
import json
ray.init(_system_config={
        "object_spilling_config": json.dumps({"type": "filesystem", "params": {"directory_path": "/tmp/spill"}})})
from variables import variables_dictionary
from models import ModelDSNN
from DataGenerator import DataGenerator
from files import training_files, validation_files
from callbacks import ParallelModelCheckpoint
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from utils import logger
import tensorflow as tf
import matplotlib.pyplot as plt
from config import config_dict, cuts
import numpy as np
from tensorflow.keras.layers.experimental import preprocessing

logger.set_log_level('INFO')

def main():

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Initialize Generators
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    training_batch_generator = DataGenerator(training_files, variables_dictionary, nbatches=250, cuts=cuts,
                                             label="Training Generator")

    validation_batch_generator = DataGenerator(validation_files, variables_dictionary, nbatches=50, cuts=cuts,
                                                label="Validation Generator")

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Initialize Model
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    # Configure model
    model_config = config_dict
    model_config["shapes"]["TauTracks"] = (len(variables_dictionary["TauTracks"]),) + (8,)
    model_config["shapes"]["ConvTrack"] = (len(variables_dictionary["ConvTrack"]),) + (4,)
    model_config["shapes"]["NeutralPFO"] = (len(variables_dictionary["NeutralPFO"]),) + (3,)
    model_config["shapes"]["ShotPFO"] = (len(variables_dictionary["ShotPFO"]),) + (8,)
    model_config["shapes"]["TauJets"] = (len(variables_dictionary["TauJets"]),)

    normalizers = {"TauTrack": preprocessing.Normalization(),
                   "NeutralPFO": preprocessing.Normalization(),
                   "ShotPFO": preprocessing.Normalization(),
                   "ConvTrack": preprocessing.Normalization(),
                   "TauJets": preprocessing.Normalization()}
    for batch in validation_batch_generator:
        normalizers["TauTrack"].adapt(batch[0][0])
        normalizers["NeutralPFO"].adapt(batch[0][1])
        normalizers["ShotPFO"].adapt(batch[0][2])
        normalizers["ConvTrack"].adapt(batch[0][3])
        normalizers["TauJets"].adapt(batch[0][4])
    training_batch_generator.reset_generator()

    model = ModelDSNN(model_config, normalizers=normalizers)

    # Configure callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss", min_delta=0.0001,
        patience=10, verbose=0, restore_best_weights=True)

    model_checkpoint = ParallelModelCheckpoint(model,
                                               path=os.path.join("data", 'weights-{epoch:02d}.h5'),
                                               monitor="val_loss", save_best_only=False, save_weights_only=True,
                                               verbose=0)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=3, min_lr=4e-6)

    callbacks = [early_stopping, model_checkpoint]#, reduce_lr]

    # Compile and summarise model
    model.summary()
    njets = 1672754.0
    n1p0n = 310223.0
    n1p1n = 729350.0
    n1pxn = 317042.0
    total = njets + n1p0n + n1p1n + n1pxn

    model.layers[-1].bias.assign([np.log(njets/(total)),
                                  np.log(n1p0n/(total)),
                                  np.log(n1p1n/(total)),
                                  np.log(n1pxn/(total)),])

    # n1p0n + n1p1n + n1pxn
    # njets + n1p1n + n1pxn
    # njets + n1p0n + n1pxn
    # njets + n1p0n + n1p1n

    weight_for_jets = (1 / njets) * (total / 2.0)
    weight_for_1p0n = (1 / n1p0n) * (total / 2.0)
    weight_for_1p1n = (1 / n1p1n) * (total / 2.0)
    weight_for_1pxn = (1 / n1pxn) * (total / 2.0)

    class_weight = {0: weight_for_jets,
                    1: weight_for_1p0n,
                    2: weight_for_1p1n,
                    3: weight_for_1pxn}


    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=[tf.keras.metrics.CategoricalAccuracy()])
    #tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
    #model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
     Train Model
     """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    history = model.fit(training_batch_generator, epochs=100, callbacks=callbacks, class_weight=class_weight,
                        validation_data=validation_batch_generator, validation_freq=1, verbose=1, shuffle=True,
                        steps_per_epoch=len(training_batch_generator))
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Make Plots 
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("plots\\loss_history.svg")
    plt.show()

    plt.plot(history.history['categorical_accuracy'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.xlabel('Epochs')
    plt.ylabel('Categorical Accuracy')
    plt.legend()
    plt.savefig("plots\\accuracy_history.svg")
    plt.show()


if __name__ == "__main__":
    main()

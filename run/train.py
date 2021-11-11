"""
Training Script
________________________________________________
Script to run the neural network training
"""

# Configure enviroment variables
import os
# Do these things first before importing
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'    # Accelerated Linear Algebra (XLA) actually seems slower
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'                # Allow tensorflow to use more GPU VRAM

import ray
import glob
import tensorflow as tf
# from tf.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

from config.variables import variables_dictionary
from scripts.DataGenerator import DataGenerator
from config.files import training_files, validation_files, ntuple_dir
from model.callbacks import ParallelModelCheckpoint
from scripts.utils import logger
from config.config import config_dict, get_cuts, models_dict
from scripts.preprocessing import Reweighter


def train(args):

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Setup enviroment
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    # Set log levels
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.tf_log_level # Sets Tensorflow Logging Level
    logger.set_log_level(args.log_level)

    # Initialize ray
    ray.init(include_dashboard=True)

    # If we're doing a learning rate scan save the models to tmp dir
    if args.run_mode == 'scan':
        args.weights_save_dir = os.path.join("network_weights", "tmp")

    # Remove old network weights
    old_weights = glob.glob(os.path.join(args.weights_save_dir, "*.h5"))
    # If we're doing a learning rate scan only remove the temporary weights file
    if args.run_mode == 'scan':
        old_weights = glob.glob(os.path.join("network_weights", "tmp", "*.h5"))
    for file in old_weights:
        os.remove(file)
    logger.log(f"Removed old weight files from {args.weights_save_dir}")
        
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Initialize Generators
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    reweighter = Reweighter(ntuple_dir, prong=args.prong)

    cuts = get_cuts(args.prong)

    training_batch_generator = DataGenerator(training_files, variables_dictionary, nbatches=100, cuts=cuts,
                                             reweighter=reweighter, prong=args.prong, label="Training Generator")

    validation_batch_generator = DataGenerator(validation_files, variables_dictionary, nbatches=50, cuts=cuts,
                                               reweighter=reweighter, prong=args.prong, label="Validation Generator")

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Initialize Model
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    # Configure model
    model_config = config_dict
    model = models_dict[args.model](model_config)

    # Configure callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0.0001,
        patience=10, verbose=0, restore_best_weights=True)

    model_checkpoint = ParallelModelCheckpoint(model, path=os.path.join(args.weights_save_dir, 'weights-{epoch:02d}.h5'),
                                               monitor="val_loss", save_best_only=True, save_weights_only=True)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=3, min_lr=4e-6)

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
    model.layers[-1].bias.assign([np.log(njets / (total)),
                                  np.log(n1p0n / (total)),
                                  np.log(n1p1n / (total)),
                                  np.log(n1pxn / (total)),
                                  np.log(n3p0n / (total)),
                                  np.log(n3pxn / (total)),
                                  ])


    opt = tf.keras.optimizers.Adam(learning_rate=args.lr) # default lr = 1e-3
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
    plt.savefig(os.path.join("plots", "loss_history.png"))
    plt.show()

    # Accuracy history
    plt.plot(history.history['categorical_accuracy'], label='train')
    plt.plot(history.history['val_categorical_accuracy'], label='val')
    plt.xlabel('Epochs')
    plt.ylabel('Categorical Accuracy')
    plt.legend()
    plt.savefig(os.path.join("plots", "accuracy_history.png"))
    plt.show()

    # Return best validation loss and accuracy
    best_val_loss_epoch = np.argmin(history.history["val_loss"])
    best_val_loss = history.history["val_loss"][best_val_loss_epoch]
    best_val_acc = history.history["val_categorical_accuracy"][best_val_loss_epoch]

    logger.log(f"Best Epoch: {best_val_loss_epoch} -- Val Loss = {best_val_loss} -- Val Acc = {best_val_acc}")

    # Shut down Ray - will raise an execption if ray.init() is called twice otherwise
    ray.shutdown()

    return best_val_loss, best_val_acc
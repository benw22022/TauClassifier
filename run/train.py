"""

Training Script
________________________________________________
Script to run the neural network training
"""

import os
import ray
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
from config.variables import variable_handler
from source.DataGenerator import DataGenerator
from config.files import training_files, validation_files, ntuple_dir
from model.callbacks import ParallelModelCheckpoint
from source.utils import logger, get_number_of_events
from config.config import config_dict, get_cuts, models_dict
from source.preprocessing import Reweighter
import shutil
import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as k
import gc
from tensorflow.keras.callbacks import Callback





def concat_datasets(datasets):
    """
    Join a list of tensorflow dataset objects into a single tf.data.Dataset
    args:
        datasets: List[tf.data.Dataset]
    returns:
        dataset: List[tf.data.Dataset]
    """
    dataset = datasets[0]
    for i in tqdm.tqdm(range(1, len(datasets))):
        dataset = dataset.concatenate(datasets[i])
    return dataset

def aux_data_filter(dataset):
    return dataset.map(lambda x, y, weights, aux: (x, y, weights))

def build_dataset(dataset_list, batch_size=32, aux_data=False):
    """
    Loads and builds a tf.data.Dataset from a number of datasets saved using 
    tf.data.experimental.save. Note: The dataset must of been orginally saved using this method
    Prefetching optimiztion is also applied to the dataset 
    args:
        filepath: str
            A file pattern that can be globbed to give a list of saved datasets
        batch_size: int
            The batch size of the data
    returns:
        dataset: tf.data.Dataset
    """
    # dataset_list = glob.glob(filepath)
    datasets = [tf.data.experimental.load(file) for file in dataset_list]
    dataset = concat_datasets(datasets)
    if not aux_data:
        dataset = aux_data_filter(dataset)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        # k.clear_session()


def train(args):

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Setup enviroment
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Set log levels
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.tf_log_level # Sets Tensorflow Logging Level
    # os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'       # Allow tensorflow to use more GPU VRAM
    ray.init()
    logger.set_log_level(args.log_level)
    if args.mixed_precision:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(policy) 

    os.system("rm -r logs/*")

    # If we're doing a learning rate scan save the models to tmp dir
    if args.run_mode == 'scan':
        args.weights_save_dir = os.path.join("network_weights", "tmp")

    old_weights = glob.glob(os.path.join(args.weights_save_dir, "*.h5"))
    # If we're doing a learning rate scan remove network weights
    if args.run_mode == 'scan':
        old_weights = glob.glob(os.path.join("network_weights", "tmp", "*.h5"))
        for file in old_weights:
            os.remove(file)
        logger.log(f"Removed old weight files from {args.weights_save_dir}")
    # Otherwise move old network weights to a backup directory (I've accidently deleted weights too many times!)
    elif len(old_weights) > 0:
        time_since_modification = os.path.getmtime(old_weights[0])
        modification_time = time.strftime('%Y-%m-%d_%H.%M.%S', time.localtime(time_since_modification))
        backup_dir = os.path.join(f"{os.path.dirname(old_weights[0])}", "backup",  modification_time)
        try:
            os.makedirs(backup_dir)
        except FileExistsError:
            pass
        for file in old_weights:
            shutil.move(file, os.path.join(backup_dir, os.path.basename(file)))
        logger.log(f"Moved old weight files to {backup_dir}")

        
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Create Datasets
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # data_files = glob.glob("data/all_data/*.dat")
    # train_files, _ = train_test_split(data_files, test_size=0.2, random_state=42)
    # train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=42)

    # train_dataset = build_dataset(train_files, batch_size=args.batch_size)
    # val_dataset = build_dataset(val_files, batch_size=10000)

    # train_dataset = build_dataset("data/train_data/*.dat", batch_size=args.batch_size)
    # val_dataset = build_dataset("data/val_data/*.dat", batch_size=10000)

    reweighter = Reweighter(ntuple_dir, prong=args.prong)

    cuts = get_cuts(args.prong)

    training_batch_generator = DataGenerator(validation_files, variable_handler, batch_size=10000, cuts=cuts,
                                             reweighter=reweighter, prong=args.prong, label="Training Generator")

    # validation_batch_generator = DataGenerator(validation_files, variable_handler, batch_size=2048, cuts=cuts,
    #                                            reweighter=reweighter, prong=args.prong, label="Validation Generator")


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Initialize Model
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    # Configure model
    logger.log(f"Creating model: {args.model}")
    model_config = config_dict
    model = models_dict[args.model](model_config)

    # Configure callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0.0001,
        patience=20, verbose=0, restore_best_weights=True)

    model_checkpoint = ParallelModelCheckpoint(model, path=os.path.join(args.weights_save_dir, 'weights-{epoch:02d}.h5'),
                                               monitor="val_loss", save_best_only=True, save_weights_only=True)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=3, min_lr=1e-9)
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
                            log_dir=log_dir,
                            histogram_freq=1,
                            write_graph=True,
                            write_images=True,
                            update_freq="epoch")

    callbacks = [early_stopping, model_checkpoint, reduce_lr, tensorboard_callback, ClearMemory()]

    # Compile and summarise model
    model.summary()

    # Following steps in: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    # Compute class weights 
    logger.log("Computing class weights", 'INFO')
    njets, n1p0n, n1p1n,  n1pxn, n3p0n, n3pxn = get_number_of_events(training_files)
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

    # Assign output layer bias
    model.layers[-1].bias.assign([np.log(njets / (total)),
                                  np.log(n1p0n / (total)),
                                  np.log(n1p1n / (total)),
                                  np.log(n1pxn / (total)),
                                  np.log(n3p0n / (total)),
                                  np.log(n3pxn / (total)),
                                  ])


    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=[tf.keras.metrics.CategoricalAccuracy()], )
    
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Train Model
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    while True:
        for i in tqdm.tqdm(range(0, len(training_batch_generator))):
            x = training_batch_generator[i]
            pass
        training_batch_generator.reset_generator()


    # history = model.fit(training_batch_generator, epochs=200, class_weight=class_weight, callbacks=callbacks,
    #                     # validation_data=validation_batch_generator, validation_freq=1, 
    #                     verbose=1, steps_per_epoch=len(training_batch_generator),
    #                     use_multiprocessing=False, workers=1)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Make Plots 
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    # Loss History
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='train')
    ax.plot(history.history['val_loss'], label='val')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig(os.path.join("plots", "loss_history.png"))
    

    # Accuracy history
    fig, ax = plt.subplots()
    ax.plot(history.history['categorical_accuracy'], label='train')
    ax.plot(history.history['val_categorical_accuracy'], label='val')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Categorical Accuracy')
    ax.legend()
    plt.savefig(os.path.join("plots", "accuracy_history.png"))

    # Return best validation loss and accuracy
    best_val_loss_epoch = np.argmin(history.history["val_loss"])
    best_val_loss = history.history["val_loss"][best_val_loss_epoch]
    best_val_acc = history.history["val_categorical_accuracy"][best_val_loss_epoch]

    logger.log(f"Best Epoch: {best_val_loss_epoch + 1} -- Val Loss = {best_val_loss} -- Val Acc = {best_val_acc}")

    return best_val_loss, best_val_acc

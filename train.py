"""

Training Script
________________________________________________
Script to run the neural network training
"""

import os
import ray
import glob
import uproot
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model.models import ModelDSNN
from source.data_generator import DataGenerator
from model.callbacks import ParallelModelCheckpoint


def get_number_of_events(files):
    all_labels = uproot.concatenate(files, filter_name="TauClassifier_Labels", library='np')["TauClassifier_Labels"]
    all_labels = np.vstack(all_labels)
    ret = []
    for l in range(0, all_labels.shape[1]):
        ret.append(sum(all_labels[:, l]))
    return ret


def train():

    ray.init()

    # Model
    model = ModelDSNN("config/model_config.yaml", "config/features.yaml")

    # Train/Test/Val
    files = glob.glob("../split_NTuples/*/*.root")
    tau_files = glob.glob("../split_NTuples/*Gammatautau*/*.root")
    tau_train_files, tau_test_files = train_test_split(tau_files, test_size=0.2, random_state=42)
    tau_train_files, tau_val_files = train_test_split(tau_train_files, test_size=0.2, random_state=42)

    jet_files = glob.glob("../split_NTuples/*JZ*/*.root")
    jet_train_files, jet_test_files = train_test_split(jet_files, test_size=0.2, random_state=42)
    jet_train_files, jet_val_files = train_test_split(jet_train_files, test_size=0.2, random_state=42)

    njets, n1p0n, n1p1n,  n1pxn, n3p0n, n3pxn = get_number_of_events(files)


    # Generators
    training_generator = DataGenerator(tau_train_files, jet_train_files, "config/features.yaml", batch_size=256)
    validation_generator = DataGenerator(tau_val_files, jet_val_files, "config/features.yaml", batch_size=1024)

    # Configure callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0.0001,
        patience=20, verbose=0, restore_best_weights=True)

    model_checkpoint = ParallelModelCheckpoint(model, path=os.path.join("network_weights", 'weights-{epoch:02d}.h5'),
                                               monitor="val_loss", save_best_only=True, save_weights_only=True)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=3, min_lr=1e-9)


    callbacks = [early_stopping, model_checkpoint, reduce_lr]

    # Compile and summarise model
    model.summary()

    # Following steps in: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    # Compute class weights 
    
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


    opt = tf.keras.optimizers.Adam()
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=[tf.keras.metrics.CategoricalAccuracy()], )
    
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Train Model
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    history = model.fit(training_generator, epochs=200, 
                        class_weight=class_weight, 
                        callbacks=callbacks,
                        validation_data=validation_generator, validation_freq=1, 
                        verbose=1, steps_per_epoch=len(training_generator),
                        use_multiprocessing=False, workers=1)

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

    return best_val_loss, best_val_acc


if __name__ == "__main__":
    train()
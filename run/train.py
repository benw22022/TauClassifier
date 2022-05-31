"""

Training Script
________________________________________________
Script to run the neural network training
"""
import logger
log = logger.get_logger(__name__)
import os
import ray
import glob
import source
import run
import uproot
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from model import configure_callbacks, ModelDSNN
from typing import Tuple
import tensorflow_addons as tfa
import focal_loss



def get_number_of_events(files):
    all_labels = uproot.concatenate(files, filter_name="TauClassifier_Labels", library='np')["TauClassifier_Labels"]
    all_labels = np.vstack(all_labels)
    class_breakdown = []
    for l in range(0, all_labels.shape[1]):
        class_breakdown.append(np.sum(all_labels[:, l]))
    return class_breakdown


def train(config: DictConfig) -> Tuple[float]:

    log.info("Running training")

    # Initialise Ray
    ray.init(runtime_env={"py_modules": [source, run, logger]})

    # Model
    model = ModelDSNN(config)

    # Grab train/val files
    tau_train_files, tau_test_files, tau_val_files = source.get_files(config, "TauFiles") 
    jet_train_files, jet_test_files, jet_val_files = source.get_files(config, "FakeFiles") 
    tau_files = tau_train_files + tau_test_files + tau_val_files
    jet_files = jet_train_files + jet_test_files + jet_val_files

    # Generators
    training_generator = source.DataGenerator(tau_train_files, jet_train_files, config, batch_size=config.batch_size, step_size=config.step_size, name='TrainGen')
    validation_generator = source.DataGenerator(tau_val_files, jet_val_files, config, batch_size=config.val_batch_size, step_size=config.step_size, name='ValGen')

    # Configure callbacks
    callbacks = configure_callbacks(config, generator=validation_generator)

    # Compile and summarise model
    model.summary()

    # Following steps in: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    # Compute class weights 
    njets, n1p0n, n1p1n,  n1pxn, n3p0n, n3pxn = get_number_of_events(tau_files + jet_files)
    total = njets + n1p0n + n1p1n + n1pxn + n3p0n + n3pxn   

    weight_for_jets = (1 / njets) * (total / 2.0) / 2
    weight_for_1p0n = (1 / n1p0n) * (total / 2.0) / 1.5
    weight_for_1p1n = (1 / n1p1n) * (total / 2.0) / 1.5
    weight_for_1pxn = (1 / n1pxn) * (total / 2.0) * 2
    weight_for_3p0n = (1 / n3p0n) * (total / 2.0) 
    weight_for_3p1n = (1 / n3pxn) * (total / 2.0) * 2

    class_weight = {0: weight_for_jets,
                    1: weight_for_1p0n,
                    2: weight_for_1p1n,
                    3: weight_for_1pxn,
                    4: weight_for_3p0n,
                    5: weight_for_3p1n,
                    }

    opt = tf.keras.optimizers.Nadam(config.learning_rate)
    # loss = tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO, alpha=config.alpha, gamma=config.gamma)
    loss = focal_loss.SparseCategoricalFocalLoss(gamma=config.gamma, class_weight=list(class_weight.values()))
    
    acc_metric = tf.keras.metrics.CategoricalAccuracy()
    if config.is_sparse:
        acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    
    model.compile(optimizer=opt, loss=loss, metrics=[acc_metric], )
    
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Train Model
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    history = model.fit(training_generator, epochs=config.epochs, 
                        # class_weight=class_weight, 
                        callbacks=callbacks,
                        validation_data=validation_generator, validation_freq=1, 
                        verbose=1, steps_per_epoch=len(training_generator),
                        use_multiprocessing=False, workers=1)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Make Plots 
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    # Make plots dir
    os.makedirs("plots", exist_ok=True)

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
    ax.plot(history.history[acc_metric.name], label='train')
    ax.plot(history.history[f'val_{acc_metric.name}'], label='val')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Categorical Accuracy')
    ax.legend()
    plt.savefig(os.path.join("plots", "accuracy_history.png"))

    # Return best validation loss and accuracy
    best_val_loss_epoch = np.argmin(history.history["val_loss"])
    best_val_loss = history.history["val_loss"][best_val_loss_epoch]
    best_val_acc = history.history["val_categorical_accuracy"][best_val_loss_epoch]

    log.info(f"Best epoch was {best_val_loss_epoch}\tloss: {best_val_loss:.3f}\tAccuracy: {best_val_acc:.2f}")

    # Run model testing 
    if config.save_ntuples:
        run.evaluate(config)
        if config.make_plots:
            run.visualise(config)

    return best_val_loss, best_val_acc
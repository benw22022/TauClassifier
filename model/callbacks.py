"""
Callbacks
___________________________________________________
Functions to configure callbacks for training
"""

import logger
log = logger.get_logger(__name__)
import os
import time
import keras
import tensorflow as tf
from omegaconf import DictConfig
from typing import List
from model.conf_matrix_cb import ConfusionMatrixCallback

class LoggingCallback(keras.callbacks.Callback):
    """
    Simple callback to log log train / val metrics to logging
    Regular tensorflow printouts are not logged
    """

    time_start = time.time()
    time_epoch_start = 0

    def on_train_begin(self, logs=None):
        log.info("Starting training")

    def on_train_end(self, logs=None):
        log.info("Training stopped")

    def on_epoch_begin(self, epoch, logs=None):
        log.info(f"Start epoch {epoch + 1}")
        self.time_epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        time_epoch_taken = time.time() - self.time_epoch_start
        time_taken = time.time() - self.time_start
        time_epoch_taken = time.strftime('%H:%M:%S', time.localtime(time_epoch_taken))
        time_taken = time.strftime('%H:%M:%S', time.localtime(time_taken))

        log.info(f"Epoch {epoch + 1} took {time_epoch_taken}")
        log.info(f"End epoch {epoch + 1}: Time elapased so far = {time_taken}")
        # log.info(f"Train Loss = {logs['train_loss']}   Train Categorical Accuracy = {logs['train_categorical_accuracy']}")
        # log.info(f"Val Loss = {logs['val_loss']}   Val Categorical Accuracy = {logs['val_categorical_accuracy']}")
        for key, value in logs.items():
            log.info(f"{key}: {value}")


def configure_callbacks(config: DictConfig, **kwargs) -> List[keras.callbacks.Callback]:
    """
    Parses config files to configure callbacks
    args:
        config: DictConfig - Hydra config object
    returns:
        List[keras.callbacks.Callback] - A list of tensorflow callbacks
    """
    callbacks = []
    
    if config.callbacks.early_stopping.enabled:
        min_delta = config.callbacks.early_stopping.min_delta
        patience=config.callbacks.early_stopping.patience

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=min_delta,
            patience=patience, verbose=0, restore_best_weights=True)

        log.info("Enabling early stopping")
        callbacks.append(early_stopping)

    if config.callbacks.model_checkpoint.enabled:
        os.makedirs("network_weights")
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join("network_weights", 'weights-{epoch:02d}.h5'),
                                                    monitor="val_loss", save_best_only=True, save_weights_only=True)
        log.info("Enabling model checkpointing")                                                    
        callbacks.append(model_checkpoint)                                                

    if config.callbacks.lr_schedd.enabled:
        factor = config.callbacks.lr_schedd.factor
        patience = config.callbacks.lr_schedd.patience
        min_lr = config.callbacks.lr_schedd.min_lr

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=patience, min_lr=min_lr)
        log.info("Enabling learing rate decay")
        callbacks.append(reduce_lr)

    if config.callbacks.tensorboard.enabled:
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir="logs", histogram_freq = 1)
        log_dir = os.path.join(os.getcwd(), 'logs')
        log.info(f"Enabling tensorboard, to start run: tensorboard --logdir{log_dir}")
        callbacks.append(tensorboard_callback)

    if config.callbacks.logging.enabled:
        log.info("Enabling training logging")
        callbacks.append(LoggingCallback())

    if config.callbacks.conf_matrix.enabled:

        if not config.callbacks.tensorboard.enabled:
            log.warn("Ignoring confusion matrix callback since is tensorboard disabled")
        else:
            try:
                log.info("Enabling cm in tensorboard")
                callbacks.append(ConfusionMatrixCallback(kwargs['generator']))
            except KeyError:
                log.error("To enable confusion matrix callback you must specify a generator")

    return callbacks
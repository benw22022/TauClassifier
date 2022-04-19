"""
Callbacks
___________________________________________________
Functions to configure callbacks for training
"""
import os
import keras
import tensorflow as tf
from omegaconf import DictConfig
from typing import List

def configure_callbacks(config: DictConfig) -> List[keras.callbacks.Callback]:
    """
    Parses config files to configure callbacks
    args:
        config: DictConfig - Hydra config object
    returns:
        List[keras.callbacks.Callback] - A list of tensorflow callbacks
    """
    callbacks = []
    
    if config.early_stopping.enabled:
        min_delta = config.early_stopping.min_delta
        patience=config.early_stopping.patience

        early_stopping = tf.keras.callbacks.EarlyStopping( monitor="val_loss", min_delta=min_delta,
            patience=patience, verbose=0, restore_best_weights=True)

        callbacks.append(early_stopping)

    if config.model_checkpoint.enabled:
        os.makedirs("network_weights")
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join("network_weights", 'weights-{epoch:02d}.h5'),
                                                    monitor="val_loss", save_best_only=True, save_weights_only=True)
        callbacks.append(model_checkpoint)                                                

    if config.lr_schedd.enabled:
        factor = config.lr_schedd.factor
        patience = config.lr_schedd.patience
        min_lr = config.lr_schedd.min_lr

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=patience, min_lr=min_lr)
        callbacks.append(reduce_lr)

    if config.tensorboard.enabled:
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir="logs", histogram_freq = 1)
        callbacks.append(tensorboard_callback)

    return callbacks
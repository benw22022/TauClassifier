"""
Callback definitions
"""
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
import keras
from timeit import default_timer as timer
import os
from omegaconf import DictConfig
from typing import List

# custom callback for multi-gpu model saving
class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self, model, path, monitor='val_loss', verbose=1,
                 save_best_only=False, save_weights_only=True):
        self._model = model
        super(ParallelModelCheckpoint, self).__init__(path, monitor, verbose, save_best_only, save_weights_only)

    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self._model)


class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)


def configure_callbacks(config: DictConfig, model: tf.keras.Model) -> List[keras.callbacks.Callback]:
    
    callbacks = []
    
    # Early stopping
    if config.early_stopping.use:
        min_delta = config.early_stopping.min_delta
        patience=config.early_stopping.patience

        early_stopping = tf.keras.callbacks.EarlyStopping( monitor="val_loss", min_delta=min_delta,
            patience=patience, verbose=0, restore_best_weights=True)

        callbacks.append(early_stopping)

    if config.model_checkpoint.use:
        os.makedirs("network_weights")
        model_checkpoint = ParallelModelCheckpoint(model, path=os.path.join("network_weights", 'weights-{epoch:02d}.h5'),
                                                    monitor="val_loss", save_best_only=True, save_weights_only=True)
        callbacks.append(model_checkpoint)                                                

    if config.lr_schedd.use:
        factor = config.lr_schedd.factor
        patience = config.lr_schedd.patience
        min_lr = config.lr_schedd.min_lr

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=patience, min_lr=min_lr)

        callbacks.append(reduce_lr)

    return callbacks
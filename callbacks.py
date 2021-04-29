"""
Callback definitions
"""
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
import os

# custom callback for multi-gpu model saving
class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self, model, path, monitor='val_loss', verbose=1,
                 save_best_only=False, save_weights_only=True):
        self._model = model
        super(ParallelModelCheckpoint, self).__init__(path, monitor, verbose, save_best_only, save_weights_only)

    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self._model)



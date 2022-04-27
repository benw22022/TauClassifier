import logger
log = logger.get_logger(__name__)
import io
import os
import time
import tqdm
import keras
import source
import source.plotting_functions as pf
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from typing import List


def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    
    buf = io.BytesIO()

    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format='png')

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Use tf.image.decode_png to convert the PNG buffer
    # to a TF image. Make sure you use 4 channels.
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Use tf.expand_dims to add the batch dimension
    image = tf.expand_dims(image, 0)

    return image

class ConfusionMatrixCallback(keras.callbacks.Callback):

    def __init__(self, generator: source.DataGenerator):
        self.generator = generator

    def on_epoch_begin(self, epoch, logs):
        print(self.params)
    
    def on_epoch_end(self, epoch, logs):

        y_true = []
        y_pred = []
        weights = []

        for i in tqdm.tqdm(range(0, len(self.generator))):
            x_batch, y_batch, w_batch = self.generator[i]
            y_pred_batch = self.model.predict(x_batch)
            y_true.append(y_batch)
            y_pred.append(y_pred_batch)
            weights.append(w_batch)
        self.generator.reset()
        
        y_true = np.concatenate([y for y in y_true])
        y_pred = np.concatenate([y for y in y_pred])
        weights = np.concatenate([w for w in weights])

        cm_fig = pf.plot_confusion_matrix(y_true, y_pred, weights, saveas=False)

        cm_image = plot_to_image(cm_fig)
    
        # Log the confusion matrix as an image summary
        file_writer_cm = tf.summary.create_file_writer(os.path.join("logs", 'cm'))
        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)
        
        log.info("Plotted confusion matrix to tensorboard")
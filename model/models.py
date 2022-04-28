"""
Models
_______________________________________________________________________________________________________________________
This is basically Bowen's Tau Decay Mode Classifier with an extra branch for TauJets
"""

import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense, Concatenate, Dropout
from tensorflow.keras import Model
from model.layers import create_deepset_input, create_dense_input
from omegaconf import DictConfig


def ModelDSNN(config: DictConfig):
    """
    TODO: docstring
    """

    initializer = tf.keras.initializers.HeNormal()
    activation = 'elu'

    # Create input branches
    x_1, b_1 = create_deepset_input(config, 'TauTracks')
    x_2, b_2 = create_deepset_input(config, "NeutralPFO")
    x_3, b_3 = create_deepset_input(config, "ShotPFO")
    x_4, b_4 = create_deepset_input(config, "ConvTrack")
    x_5, b_5 = create_dense_input(config, "TauJets")
   
    # Concatenate inputs
    merged = Concatenate()([b_1, b_2, b_3, b_4, b_5])

    # Final dense layers
    for n in config.n_dense_merged:
        merged = Dense(n, kernel_initializer=initializer)(merged)
        merged = Activation(activation)(merged)
        merged = Dropout(config.dropout)(merged)

    y = Dense(config.n_classes, activation="softmax")(merged)

    return Model(inputs=[x_1, x_2, x_3, x_4, x_5], outputs=y)
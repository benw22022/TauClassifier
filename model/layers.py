"""
Layers
_________________________________________________________
Functions to create custom layers
"""

"""
Layers
_________________________________________________________________________
Function and class definitions for model layers and model componants
"""

import yaml
from keras import backend as kbe
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Masking, TimeDistributed, Concatenate
from tensorflow.keras.layers import Layer, Activation, BatchNormalization, MultiHeadAttention, LayerNormalization
from tensorflow.keras import Model
from model.set_transformer.model import BasicSetTransformer
from omegaconf import DictConfig
# from model.set_transformer.blocks import *

class Sum(Layer):
    """Simple sum layer.

    The tricky bits are getting masking to work properly, but given
    that time distributed dense layers _should_ compute masking on
    their own

    See Dan's implemention:
    https://gitlab.cern.ch/deep-sets-example/higgs-regression-training/blob/master/SumLayer.py

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        if mask is not None:
            x = x * kbe.cast(mask, kbe.dtype(x))[:, :, None]
        return kbe.sum(x, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

    def compute_mask(self, inputs, mask=None):
        return None

def create_deepset_input(config: DictConfig, branchname: str, activation: str='elu', initializer: tf.keras.initializers.Initializer=tf.keras.initializers.HeNormal(),
                         regularizer: tf.keras.regularizers.Regularizer=None, self_attention=False):
    """
    Create a Deepset input branch
    args:
        config: DictConfig - Hydra config object
        branchname: str - Name of the input in the config
        activation: str - Name of compatible activation function
        initializer: tf.keras.initializers.Initializer=tf.keras.initializers.HeNormal() - Kernal initializer function
    returns:
        input_layer, dense_layer
    """
    
    # Input
    input_layer = Input(shape=(len(config.branches[branchname].features), config.branches[branchname].max_objects), name=f'input_{branchname}')
    tdd_layer = Masking(mask_value=config.mask_value, name=f'masked_{branchname}_input')(input_layer)

    # Time Distributed layers
    for i, n in enumerate(config.n_inputs[branchname]):
        tdd_layer = TimeDistributed(Dense(n, kernel_initializer=initializer, kernel_regularizer=regularizer), name=f'tdd_{branchname}_{i}-{n}')(tdd_layer)
        if self_attention:
            # tdd_layer = MultiHeadAttention(config.branches[branchname].max_objects, config.branches[branchname].max_objects)(tdd_layer, tdd_layer)
            tdd_layer = MultiHeadAttention(num_heads=len(config.branches[branchname].features), key_dim=config.branches[branchname].max_objects, name=f"self_attention_{branchname}_{i}")(tdd_layer, tdd_layer)
        tdd_layer = Activation(activation, name=f"tdd_{branchname}_activation_{i}")(tdd_layer)
        if config.batch_norm:
            # tdd_layer = BatchNormalization(name=f"batchnorm_tdd_{branchname}_{i}")(tdd_layer)
            tdd_layer = LayerNormalization(name=f"layernorm_tdd_{branchname}_{i}")(tdd_layer)
            
    
    
    # Deepset Sum Layer
    dense_layer = Sum(name=f'sum_{branchname}')(tdd_layer)

    # Regular dense layers
    for i, n in enumerate(config.n_hiddens[branchname]):
        dense_layer = Dense(n, kernel_initializer=initializer, kernel_regularizer=regularizer, name=f'dense_{branchname}_{i}-{n}')(dense_layer)
        dense_layer = Activation(activation, name=f"dense_{branchname}_activation_{i}")(dense_layer)
        if config.batch_norm:
            dense_layer = BatchNormalization(name=f"batchnorm_dense_{branchname}_{i}")(dense_layer)
        
    return input_layer, dense_layer

def create_dense_input(config: DictConfig, branchname: str, activation: str='elu', initializer: tf.keras.initializers.Initializer=tf.keras.initializers.HeNormal(),
                       regularizer: tf.keras.regularizers.Regularizer=None):
    """
    Create a dense network input branch
    args:
        config: DictConfig - Hydra config object
        branchname: str - Name of the input in the config
        activation: str - Name of compatible activation function
        initializer: tf.keras.initializers.Initializer=tf.keras.initializers.HeNormal() - Kernal initializer function
    returns:
        input_layer, dense_layer
    """
    
    input_layer = Input(shape=(len(config.branches[branchname].features)), name=f'input_{branchname}')
    dense_layer = input_layer
    for i, n in enumerate(config.n_hiddens[branchname]):
        dense_layer = Dense(n, activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer, name=f'dense_{branchname}_{i}-{n}')(dense_layer)
        if config.batch_norm:
            dense_layer = BatchNormalization(name=f"batchnorm_dense_{branchname}_{i}")(dense_layer)
    
    return input_layer, dense_layer



# def create_attn_deepset_input(config: DictConfig, branchname: str, activation: str='elu', initializer: tf.keras.initializers.Initializer=tf.keras.initializers.HeNormal(),
#                          regularizer: tf.keras.regularizers.Regularizer=None):
#     """
#     Create a Deepset input branch
#     args:
#         config: DictConfig - Hydra config object
#         branchname: str - Name of the input in the config
#         activation: str - Name of compatible activation function
#         initializer: tf.keras.initializers.Initializer=tf.keras.initializers.HeNormal() - Kernal initializer function
#     returns:
#         input_layer, dense_layer
#     """
#     
#     # Input
#     input_layer = Input(shape=(len(config.branches[branchname].features), config.branches[branchname].max_objects), name=f'input_{branchname}')
#     tdd_layer = Masking(mask_value=config.mask_value, name=f'masked_{branchname}_input')(input_layer)
# 
#     #  # Time Distributed layers
#     # for i, n in enumerate(config.n_inputs[branchname]):
#     #     tdd_layer = TimeDistributed(Dense(n, kernel_initializer=initializer, kernel_regularizer=regularizer), name=f'tdd_{branchname}_{i}-{n}')(tdd_layer)
#     #     # tdd_layer = MultiHeadAttention(num_heads=config.branches[branchname].max_objects, key_dim=config.branches[branchname].max_objects)(tdd_layer, tdd_layer)
#     #     tdd_layer = Activation(activation, name=f"tdd_{branchname}_activation_{i}")(tdd_layer)
#     #     if config.batch_norm:
#     #         tdd_layer = BatchNormalization(name=f"batchnorm_tdd_{branchname}_{i}")(tdd_layer)
#     
#     max_objs = config.branches[branchname].max_objects
#     
#     dense_layer = BasicSetTransformer(out_dim=20)(tdd_layer)
#     # dense_layer = Dense(config.n_hiddens[branchname][0], kernel_initializer=initializer, kernel_regularizer=regularizer, name=f'dense_{branchname}_{i}-{n}')(tdd_layer)
#     
#     
#     # Deepset Sum Layer
#     # dense_layer = Sum(name=f'sum_{branchname}')(tdd_layer)
# 
#     # Regular dense layers
#     # for i, n in enumerate(config.n_hiddens[branchname]):
#     #     dense_layer = Dense(n, kernel_initializer=initializer, kernel_regularizer=regularizer, name=f'dense_{branchname}_{i}-{n}')(dense_layer)
#     #     dense_layer = Activation(activation, name=f"dense_{branchname}_activation_{i}")(dense_layer)
#     #     if config.batch_norm:
#     #         dense_layer = BatchNormalization(name=f"batchnorm_dense_{branchname}_{i}")(dense_layer)
#         
#     return input_layer, dense_layer
# 
# 
# def create_simple_attn_deepset_input(config: DictConfig, branchname: str, activation: str='elu', initializer: tf.keras.initializers.Initializer=tf.keras.initializers.HeNormal(),
#                          regularizer: tf.keras.regularizers.Regularizer=None):
#     """
#     Create a Deepset input branch
#     args:
#         config: DictConfig - Hydra config object
#         branchname: str - Name of the input in the config
#         activation: str - Name of compatible activation function
#         initializer: tf.keras.initializers.Initializer=tf.keras.initializers.HeNormal() - Kernal initializer function
#     returns:
#         input_layer, dense_layer
#     """
#     
#     # Input
#     input_layer = Input(shape=(len(config.branches[branchname].features), config.branches[branchname].max_objects), name=f'input_{branchname}')
#     tdd_layer = Masking(mask_value=config.mask_value, name=f'masked_{branchname}_input')(input_layer)
# 
#      # Time Distributed layers
#     for i, n in enumerate(config.n_inputs[branchname]):
#         tdd_layer = TimeDistributed(Dense(n, kernel_initializer=initializer, kernel_regularizer=regularizer), name=f'tdd_{branchname}_{i}-{n}')(tdd_layer)
#         tdd_layer = MultiHeadAttention(num_heads=config.branches[branchname].max_objects, key_dim=config.branches[branchname].max_objects)(tdd_layer, tdd_layer)
#         tdd_layer = Activation(activation, name=f"tdd_{branchname}_activation_{i}")(tdd_layer)
#         if config.batch_norm:
#             tdd_layer = BatchNormalization(name=f"batchnorm_tdd_{branchname}_{i}")(tdd_layer)
#     
#     
#     # Deepset Sum Layer
#     dense_layer = Sum(name=f'sum_{branchname}')(tdd_layer)
# 
#     # Regular dense layers
#     for i, n in enumerate(config.n_hiddens[branchname]):
#         dense_layer = Dense(n, kernel_initializer=initializer, kernel_regularizer=regularizer, name=f'dense_{branchname}_{i}-{n}')(dense_layer)
#         dense_layer = Activation(activation, name=f"dense_{branchname}_activation_{i}")(dense_layer)
#         if config.batch_norm:
#             dense_layer = BatchNormalization(name=f"batchnorm_dense_{branchname}_{i}")(dense_layer)
#         
#     return input_layer, dense_layer
"""
Models
"""

from keras.models import Model
from keras.layers import Input, Dense, LSTM, Masking, TimeDistributed, concatenate

import yaml
from keras import backend as kbe
from keras.layers import Input, Dense, LSTM, Masking, TimeDistributed, Concatenate, Bidirectional
from keras.layers import Layer, Activation, BatchNormalization, LeakyReLU, Dropout
from keras.models import Model
from tensorflow.keras import layers as tfk_layers
# for keras tuner
from tensorflow.keras import models as tfk_models


# =============
# Custom Layers
# =============

class Sum(Layer):
    """Simple sum layer.

    The tricky bits are getting masking to work properly, but given
    that time distributed dense layers _should_ compute masking on
    their own

    See Dan's implementation:
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


class SumTFK(tfk_layers.Layer):
    """Simple sum layer.

    The tricky bits are getting masking to work properly, but given
    that time distributed dense layers _should_ compute masking on
    their own

    See Dan's implementation:
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


# =================
# Functional models
# =================

def ModelDSNN(para, mask_value=0.0):
    #para = yaml.full_load(open(config_file))["DSNN"]
    bn = True #if para["batch_norm"] == 1 else False

    # Branch 1
    x_1 = Input(shape=para["shapes"]["TauTrack"])
    b_1 = Masking(mask_value=mask_value)(x_1)
    for x in range(para["n_tdd"]["TauTrack"]):
        b_1 = TimeDistributed(Dense(para["n_inputs"]["TauTrack"][x]))(b_1)
        b_1 = Activation("relu")(b_1)
    b_1 = Sum()(b_1)
    for x in range(para["n_h"]["TauTrack"]):
        b_1 = Dense(para["n_hiddens"]["TauTrack"][x])(b_1)
        b_1 = Activation("relu")(b_1)
    if bn:
        b_1 = BatchNormalization()(b_1)

    # Branch 2
    x_2 = Input(shape=para["shapes"]["NeutralPFO"])
    b_2 = Masking(mask_value=mask_value)(x_2)
    for x in range(para["n_tdd"]["NeutralPFO"]):
       b_2 = TimeDistributed(Dense(para["n_inputs"]["NeutralPFO"][x]))(b_2)
       b_2 = Activation("relu")(b_2)
    b_2 = Sum()(b_2)
    for x in range(para["n_h"]["NeutralPFO"]):
       b_2 = Dense(para["n_hiddens"]["NeutralPFO"][x])(b_2)
       b_2 = Activation("relu")(b_2)
    if bn:
       b_2 = BatchNormalization()(b_2)

    # Branch 3
    x_3 = Input(shape=para["shapes"]["ShotPFO"])
    b_3 = Masking(mask_value=mask_value)(x_3)
    for x in range(para["n_tdd"]["ShotPFO"]):
        b_3 = TimeDistributed(Dense(para["n_inputs"]["ShotPFO"][x]))(b_3)
        b_3 = Activation("relu")(b_3)
    b_3 = Sum()(b_3)
    for x in range(para["n_h"]["ShotPFO"]):
        b_3 = Dense(para["n_hiddens"]["ShotPFO"][x])(b_3)
        b_3 = Activation("relu")(b_3)
    if bn:
        b_3 = BatchNormalization()(b_3)

    # Branch 4
    x_4 = Input(shape=para["shapes"]["ConvTrack"])
    b_4 = Masking(mask_value=mask_value)(x_4)
    for x in range(para["n_tdd"]["ConvTrack"]):
        b_4 = TimeDistributed(Dense(para["n_inputs"]["ConvTrack"][x]))(b_4)
        b_4 = Activation("relu")(b_4)
    b_4 = Sum()(b_4)
    for x in range(para["n_h"]["ConvTrack"]):
        b_4 = Dense(para["n_hiddens"]["ConvTrack"][x])(b_4)
        b_4 = Activation("relu")(b_4)
    if bn:
        b_4 = BatchNormalization()(b_4)

    # Branch 5
    x_5 = Input(shape=para["shapes"]["TauJets"])
    dense_5_1 = Dense(20, activation="relu")(x_5)
    dense_5_2 = Dense(10, activation="relu")(dense_5_1)
    dense_5_3 = Dense(5, activation="relu")(dense_5_2)
    BatchNormalization()(x_5)

    # Merge
    merged = Concatenate()([b_1, b_2, b_3, b_4, dense_5_3])
    #merged = Dropout(para["dropout"])(merged)
    merged = Dense(para["n_fc1"])(merged)
    merged = Activation("relu")(merged)
    #merged = Dropout(para["dropout"])(merged)
    merged = Dense(para["n_fc2"])(merged)
    merged = Activation("relu")(merged)

    #y = Dense(para["n_classes"], activation="softmax")(merged)
    y = Dense(1, activation="sigmoid")(merged)

    return Model(inputs=[x_1, x_2, x_3, x_4, x_5], outputs=y)


def tauid_rnn_model(
        input_shape_1, input_shape_2, input_shape_3,
        dense_units_1_1=32, dense_units_1_2=32,
        lstm_units_1_1=32, lstm_units_1_2=32,
        dense_units_2_1=32, dense_units_2_2=32,
        lstm_units_2_1=32, lstm_units_2_2=32,
        dense_units_3_1=128, dense_units_3_2=128, dense_units_3_3=16,
        merge_dense_units_1=64, merge_dense_units_2=32,
        backwards=False, mask_value=0.0, unroll=True, incl_clusters=True):
    """
    TODO: Documentation
    """
    # Branch 1
    x_1 = Input(shape=input_shape_1)
    mask_1 = Masking(mask_value=mask_value)(x_1)
    shared_dense_1_1 = TimeDistributed(
        Dense(dense_units_1_1, activation="relu"))(mask_1)
    shared_dense_1_2 = TimeDistributed(
        Dense(dense_units_1_2, activation="relu"))(shared_dense_1_1)
    lstm_1_1 = LSTM(lstm_units_1_1, unroll=unroll, go_backwards=backwards,
                    activation="relu", return_sequences=True)(shared_dense_1_2)
    lstm_1_2 = LSTM(lstm_units_1_2, unroll=unroll, go_backwards=backwards,
                    activation="relu")(lstm_1_1)

    # Branch 2
    if incl_clusters:
        x_2 = Input(shape=input_shape_2)
        mask_2 = Masking(mask_value=mask_value)(x_2)
        shared_dense_2_1 = TimeDistributed(
            Dense(dense_units_2_1, activation="relu"))(mask_2)
        shared_dense_2_2 = TimeDistributed(
            Dense(dense_units_2_2, activation="relu"))(shared_dense_2_1)
        lstm_2_1 = LSTM(lstm_units_2_1, unroll=unroll, go_backwards=backwards,
                        activation="relu", return_sequences=True)(shared_dense_2_2)
        lstm_2_2 = LSTM(lstm_units_2_2, unroll=unroll, go_backwards=backwards,
                        activation="relu")(lstm_2_1)

    # Branch 3
    x_3 = Input(shape=input_shape_3)
    dense_3_1 = Dense(dense_units_3_1, activation="relu")(x_3)
    dense_3_2 = Dense(dense_units_3_2, activation="relu")(dense_3_1)
    dense_3_3 = Dense(dense_units_3_3, activation="relu")(dense_3_2)

    # Merge
    if incl_clusters:
        merged_branches = concatenate([lstm_1_2, lstm_2_2, dense_3_3])
    else:
        merged_branches = concatenate([lstm_1_2, dense_3_3])

    merge_dense_1 = Dense(merge_dense_units_1, activation="relu")(
        merged_branches)
    merge_dense_2 = Dense(merge_dense_units_2, activation="relu")(
        merge_dense_1)
    y = Dense(1, activation="sigmoid")(merge_dense_2)

    if incl_clusters:
        return Model(inputs=[x_1, x_2, x_3], outputs=y)
    else:
        return Model(inputs=[x_1, x_3], outputs=y)



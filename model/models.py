"""
Models
_______________________________________________________________________________________________________________________
This is basically Bowen's Tau Decay Mode Classifier with an extra branch for TauJets
"""


from keras import backend as kbe
from tensorflow.keras.layers import Input, Dense, Masking, TimeDistributed, Concatenate
from tensorflow.keras.layers import Layer, Activation, BatchNormalization
from tensorflow.keras import Model
import tensorflow as tf
from model.set_transformer.model import BasicSetTransformer

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


# =================
# Functional models
# =================

def ModelDSNN(para, mask_value=-1, normalizers=None, bn=True):
    """
    TODO: docstring
    """
    initializer = tf.keras.initializers.HeNormal()
    activation_func = 'relu'

    # Branch 1
    x_1 = Input(shape=para["shapes"]["TauTrack"])
    b_1 = Masking(mask_value=mask_value)(x_1)
    # if normalizers is not None:
    #     b_1 = normalizers["TauTrack"](b_1)
    for x in range(para["n_tdd"]["TauTrack"]):
        b_1 = TimeDistributed(Dense(para["n_inputs"]["TauTrack"][x], kernel_initializer=initializer))(b_1)
        b_1 = Activation(activation_func)(b_1)
    b_1 = Sum()(b_1)
    for x in range(para["n_h"]["TauTrack"]):
        b_1 = Dense(para["n_hiddens"]["TauTrack"][x], kernel_initializer=initializer)(b_1)
        b_1 = Activation(activation_func)(b_1)
    # if bn:
    #     b_1 = BatchNormalization()(b_1)

    # Branch 2
    x_2 = Input(shape=para["shapes"]["NeutralPFO"])
    b_2 = Masking(mask_value=mask_value)(x_2)
    # if normalizers is not None:
    #     b_2 = normalizers["NeutralPFO"](b_2)
    for x in range(para["n_tdd"]["NeutralPFO"]):
       b_2 = TimeDistributed(Dense(para["n_inputs"]["NeutralPFO"][x], kernel_initializer=initializer))(b_2)
       b_2 = Activation(activation_func)(b_2)
    b_2 = Sum()(b_2)
    for x in range(para["n_h"]["NeutralPFO"]):
       b_2 = Dense(para["n_hiddens"]["NeutralPFO"][x], kernel_initializer=initializer)(b_2)
       b_2 = Activation(activation_func)(b_2)
    # if bn:
    #    b_2 = BatchNormalization()(b_2)

    # Branch 3
    x_3 = Input(shape=para["shapes"]["ShotPFO"])
    b_3 = Masking(mask_value=mask_value)(x_3)
    # if normalizers is not None:
    #     b_3 = normalizers["ShotPFO"](b_3)
    for x in range(para["n_tdd"]["ShotPFO"]):
        b_3 = TimeDistributed(Dense(para["n_inputs"]["ShotPFO"][x], kernel_initializer=initializer))(b_3)
        b_3 = Activation(activation_func)(b_3)
    b_3 = Sum()(b_3)
    for x in range(para["n_h"]["ShotPFO"]):
        b_3 = Dense(para["n_hiddens"]["ShotPFO"][x], kernel_initializer=initializer)(b_3)
        b_3 = Activation(activation_func)(b_3)
    # if bn:
    #     b_3 = BatchNormalization()(b_3)

    # Branch 4
    x_4 = Input(shape=para["shapes"]["ConvTrack"])
    b_4 = Masking(mask_value=mask_value)(x_4)
    # if normalizers is not None:
    #     b_4 = normalizers["ConvTrack"](b_4)
    for x in range(para["n_tdd"]["ConvTrack"]):
        b_4 = TimeDistributed(Dense(para["n_inputs"]["ConvTrack"][x], kernel_initializer=initializer))(b_4)
        b_4 = Activation(activation_func)(b_4)
    b_4 = Sum()(b_4)
    for x in range(para["n_h"]["ConvTrack"]):
        b_4 = Dense(para["n_hiddens"]["ConvTrack"][x], kernel_initializer=initializer)(b_4)
        b_4 = Activation(activation_func)(b_4)
    # if bn:
    #     b_4 = BatchNormalization()(b_4)

    # Branch 5
    x_5 = Input(shape=para["shapes"]["TauJets"])
    b_5 = x_5
    # if normalizers is not None:
    #     b_5 = normalizers["TauJets"](b_5)
    b_5 = Dense(60, activation=activation_func, kernel_initializer=initializer)(b_5)
    b_5 = Dense(30, activation=activation_func, kernel_initializer=initializer)(b_5)
    b_5 = Dense(15, activation=activation_func, kernel_initializer=initializer)(b_5)
    # if bn:
    #     b_5 = BatchNormalization()(b_5)

    # Merge
    merged = Concatenate()([b_1, b_2, b_3, b_4, b_5])
    merged = Dense(para["n_fc1"], kernel_initializer=initializer)(merged)
    merged = Activation(activation_func)(merged)
    #merged = Dropout(para["dropout"])(merged)
    merged = Dense(para["n_fc2"], kernel_initializer=initializer)(merged)
    merged = Activation(activation_func)(merged)

    # y = Dense(para["n_classes"], activation="softmax")(merged)
    y = Dense(6, activation="softmax")(merged)

    return Model(inputs=[x_1, x_2, x_3, x_4, x_5], outputs=y)


def ModelDSNN_2Step(para, mask_value=-1, normalizers=None, bn=False):
    """
    Crazy idea I read in here https://arxiv.org/pdf/1601.01157.pdf
    Basically a 2 step network where you feed the predictions from the 1st pass into the 2nd
    """
    initializer = tf.keras.initializers.HeNormal()
    activation_func = 'swish'

    # Branch 1
    x_1 = Input(shape=para["shapes"]["TauTrack"])
    b_1 = Masking(mask_value=mask_value)(x_1)
    if normalizers is not None:
        b_1 = normalizers["TauTrack"](b_1)
    for x in range(para["n_tdd"]["TauTrack"]):
        b_1 = TimeDistributed(Dense(para["n_inputs"]["TauTrack"][x], kernel_initializer=initializer))(b_1)
        b_1 = Activation(activation_func)(b_1)
    b_1 = Sum()(b_1)
    for x in range(para["n_h"]["TauTrack"]):
        b_1 = Dense(para["n_hiddens"]["TauTrack"][x], kernel_initializer=initializer)(b_1)
        b_1 = Activation(activation_func)(b_1)
    if bn:
        b_1 = BatchNormalization()(b_1)

    # Branch 2
    x_2 = Input(shape=para["shapes"]["NeutralPFO"])
    b_2 = Masking(mask_value=mask_value)(x_2)
    if normalizers is not None:
        b_2 = normalizers["NeutralPFO"](b_2)
    for x in range(para["n_tdd"]["NeutralPFO"]):
       b_2 = TimeDistributed(Dense(para["n_inputs"]["NeutralPFO"][x], kernel_initializer=initializer))(b_2)
       b_2 = Activation(activation_func)(b_2)
    b_2 = Sum()(b_2)
    for x in range(para["n_h"]["NeutralPFO"]):
       b_2 = Dense(para["n_hiddens"]["NeutralPFO"][x], kernel_initializer=initializer)(b_2)
       b_2 = Activation(activation_func)(b_2)
    if bn:
       b_2 = BatchNormalization()(b_2)

    # Branch 3
    x_3 = Input(shape=para["shapes"]["ShotPFO"])
    b_3 = Masking(mask_value=mask_value)(x_3)
    if normalizers is not None:
        b_3 = normalizers["ShotPFO"](b_3)
    for x in range(para["n_tdd"]["ShotPFO"]):
        b_3 = TimeDistributed(Dense(para["n_inputs"]["ShotPFO"][x], kernel_initializer=initializer))(b_3)
        b_3 = Activation(activation_func)(b_3)
    b_3 = Sum()(b_3)
    for x in range(para["n_h"]["ShotPFO"]):
        b_3 = Dense(para["n_hiddens"]["ShotPFO"][x], kernel_initializer=initializer)(b_3)
        b_3 = Activation(activation_func)(b_3)
    if bn:
        b_3 = BatchNormalization()(b_3)

    # Branch 4
    x_4 = Input(shape=para["shapes"]["ConvTrack"])
    b_4 = Masking(mask_value=mask_value)(x_4)
    if normalizers is not None:
        b_4 = normalizers["ConvTrack"](b_4)
    for x in range(para["n_tdd"]["ConvTrack"]):
        b_4 = TimeDistributed(Dense(para["n_inputs"]["ConvTrack"][x], kernel_initializer=initializer))(b_4)
        b_4 = Activation(activation_func)(b_4)
    b_4 = Sum()(b_4)
    for x in range(para["n_h"]["ConvTrack"]):
        b_4 = Dense(para["n_hiddens"]["ConvTrack"][x], kernel_initializer=initializer)(b_4)
        b_4 = Activation(activation_func)(b_4)
    if bn:
        b_4 = BatchNormalization()(b_4)

    # Branch 5
    x_5 = Input(shape=para["shapes"]["TauJets"])
    b_5 = x_5
    if normalizers is not None:
        b_5 = normalizers["TauJets"](b_5)
    b_5 = Dense(60, activation=activation_func, kernel_initializer=initializer)(b_5)
    b_5 = Dense(30, activation=activation_func, kernel_initializer=initializer)(b_5)
    b_5 = Dense(15, activation=activation_func, kernel_initializer=initializer)(b_5)
    if bn:
        b_5 = BatchNormalization()(b_5)

    # Merge
    merged = Concatenate()([b_1, b_2, b_3, b_4, b_5])
    merged = Dense(para["n_fc1"], kernel_initializer=initializer)(merged)
    merged = Activation(activation_func)(merged)
    #merged = Dropout(para["dropout"])(merged)
    merged = Dense(para["n_fc2"], kernel_initializer=initializer)(merged)
    merged = Activation(activation_func)(merged)

    y1 = Dense(6, activation="softmax")(merged)

    # Branch 1
    x_12 = x_1
    b_12 = Masking(mask_value=mask_value)(x_12)
    if normalizers is not None:
        b_12 = normalizers["TauTrack"](b_12)
    for x in range(para["n_tdd"]["TauTrack"]):
        b_12 = TimeDistributed(Dense(para["n_inputs"]["TauTrack"][x], kernel_initializer=initializer))(b_12)
        b_12 = Activation(activation_func)(b_12)
    b_12 = Sum()(b_12)
    for x in range(para["n_h"]["TauTrack"]):
        b_12 = Dense(para["n_hiddens"]["TauTrack"][x], kernel_initializer=initializer)(b_12)
        b_12 = Activation(activation_func)(b_12)
    if bn:
        b_12 = BatchNormalization()(b_12)

    # Branch 2
    x_22 = x_2
    b_22 = Masking(mask_value=mask_value)(x_22)
    if normalizers is not None:
        b_22 = normalizers["NeutralPFO"](b_22)
    for x in range(para["n_tdd"]["NeutralPFO"]):
       b_22 = TimeDistributed(Dense(para["n_inputs"]["NeutralPFO"][x], kernel_initializer=initializer))(b_22)
       b_22 = Activation(activation_func)(b_22)
    b_22 = Sum()(b_22)
    for x in range(para["n_h"]["NeutralPFO"]):
       b_22 = Dense(para["n_hiddens"]["NeutralPFO"][x], kernel_initializer=initializer)(b_22)
       b_22 = Activation(activation_func)(b_22)
    if bn:
       b_22 = BatchNormalization()(b_22)

    # Branch 3
    x_32 = x_3
    b_32 = Masking(mask_value=mask_value)(x_32)
    if normalizers is not None:
        b_32 = normalizers["ShotPFO"](b_32)
    for x in range(para["n_tdd"]["ShotPFO"]):
        b_32 = TimeDistributed(Dense(para["n_inputs"]["ShotPFO"][x], kernel_initializer=initializer))(b_32)
        b_32 = Activation(activation_func)(b_32)
    b_32 = Sum()(b_32)
    for x in range(para["n_h"]["ShotPFO"]):
        b_32 = Dense(para["n_hiddens"]["ShotPFO"][x], kernel_initializer=initializer)(b_32)
        b_32 = Activation(activation_func)(b_32)
    if bn:
        b_32 = BatchNormalization()(b_32)

    # Branch 4
    x_42 = x_4
    b_42 = Masking(mask_value=mask_value)(x_42)
    if normalizers is not None:
        b_42 = normalizers["ConvTrack"](b_42)
    for x in range(para["n_tdd"]["ConvTrack"]):
        b_42 = TimeDistributed(Dense(para["n_inputs"]["ConvTrack"][x], kernel_initializer=initializer))(b_42)
        b_42 = Activation(activation_func)(b_42)
    b_42 = Sum()(b_42)
    for x in range(para["n_h"]["ConvTrack"]):
        b_42 = Dense(para["n_hiddens"]["ConvTrack"][x], kernel_initializer=initializer)(b_42)
        b_42 = Activation(activation_func)(b_42)
    if bn:
        b_42 = BatchNormalization()(b_42)

    # Branch 5
    x_52 = x_5
    b_52 = x_52
    if normalizers is not None:
        b_52 = normalizers["TauJets"](b_52)
    b_52 = Dense(60, activation=activation_func, kernel_initializer=initializer)(b_52)
    b_52 = Dense(30, activation=activation_func, kernel_initializer=initializer)(b_52)
    b_52 = Dense(15, activation=activation_func, kernel_initializer=initializer)(b_52)
    if bn:
        b_52 = BatchNormalization()(b_5)

    # Merge
    merged2 = Concatenate()([b_12, b_22, b_32, b_42, b_52, y1])
    merged2 = Dense(para["n_fc1"], kernel_initializer=initializer)(merged2)
    merged2 = Activation(activation_func)(merged2)
    #merged = Dropout(para["dropout"])(merged)
    merged2 = Dense(para["n_fc2"], kernel_initializer=initializer)(merged2)
    merged2 = Activation(activation_func)(merged2)

    y2 = Dense(6, activation="softmax")(merged2)

    return Model(inputs=[x_1, x_2, x_3, x_4, x_5], outputs=y2)




def SetTransformer(para, mask_value=-4.0,):
    """
    SetTransformer implementation in TensorFlow by https://github.com/arrigonialberto86/set_transformer
    Based on this paper https://arxiv.org/abs/1810.00825
    Only had limited time to play with this
    Couldn't really get this to work and I've not had the time and resources to properly understand it
    If you're reading this perhaps give it a go?
    """
    initializer = tf.keras.initializers.HeNormal()
    activation_func = 'elu'
    out_dim = 25

    # Branch 1
    x_1 = Input(shape=para["shapes"]["TauTrack"], ragged=True)
    b_1 = Masking(mask_value=mask_value)(x_1)
    b_1 = BasicSetTransformer(out_dim=out_dim)(b_1)
    # b_1 = BatchNormalization()(b_1)

    # Branch 2
    x_2 = Input(shape=para["shapes"]["NeutralPFO"], ragged=True)
    b_2 = Masking(mask_value=mask_value)(x_2)
    b_2 = BasicSetTransformer(out_dim=out_dim)(b_2)
    # b_2 = BatchNormalization()(b_2)

    # Branch 3
    x_3 = Input(shape=para["shapes"]["ShotPFO"], ragged=True)
    b_3 = Masking(mask_value=mask_value)(x_3)
    b_3 = BasicSetTransformer(out_dim=out_dim)(b_3)
    # b_3 = BatchNormalization()(b_3)

    # Branch 4
    x_4 = Input(shape=para["shapes"]["ConvTrack"], ragged=True)
    b_4 = Masking(mask_value=mask_value)(x_4)
    b_4 = BasicSetTransformer(out_dim=out_dim)(b_4)
    # b_4 = BatchNormalization()(b_4)

    # Branch 5
    x_5 = Input(shape=para["shapes"]["TauJets"])
    b_5 = x_5
    b_5 = Dense(30, activation=activation_func, kernel_initializer=initializer)(b_5)
    b_5 = Dense(15, activation=activation_func, kernel_initializer=initializer)(b_5)
    b_5 = Dense(10, activation=activation_func, kernel_initializer=initializer)(b_5)
    b_5 = BatchNormalization()(b_5)

    # Merge
    merged = Concatenate()([b_1, b_2, b_3, b_4, b_5])
    merged = Dense(para["n_fc1"], kernel_initializer=initializer)(merged)
    merged = Activation(activation_func)(merged)
    #merged = Dropout(para["dropout"])(merged)
    merged = Dense(para["n_fc2"], kernel_initializer=initializer)(merged)
    merged = Activation(activation_func)(merged)

    #y = Dense(para["n_classes"], activation="softmax")(merged)
    y = Dense(6, activation="softmax")(merged)

    return Model(inputs=[x_1, x_2, x_3, x_4, x_5], outputs=y)
    #return Model(inputs=[x_5], outputs=y)

"""
Models
"""

from keras.models import Model
from keras.layers import Input, Dense, LSTM, Masking, TimeDistributed, concatenate


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



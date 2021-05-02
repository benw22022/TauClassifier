"""
Main Code Body
"""
import os
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'    # Acclerated Linear Algbra (XLA) Seems to actually make things slower
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                     # Disables GPU

from variables import variables_dictionary
from models import ModelDSNN
from DataGenerator import DataGenerator
from files import training_files_dictionary, validation_files_dictionary
from callbacks import ParallelModelCheckpoint, TimingCallback
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from utils import logger
import tensorflow as tf
from config import config_dict
logger.set_log_level('INFO')


def main():
    logger.log("Beginning dataset preparation", 'INFO')

    # Initialize Generators
    cuts = {"Gammatautau": "TauJets.truthProng == 1"}
    training_batch_generator = DataGenerator(training_files_dictionary, variables_dictionary, nbatches=250, cuts=cuts,
                                             label="Training Generator")

    # Work out input shapes
    # shape_trk, shape_neut_pfo, shape_shot_pfo, shape_conv_trk, shape_jet, shape_label, shape_weight = training_batch_generator.get_batch_shapes()
    # training_batch_generator.reset_generator()
    # logger.log(f"TauTracks Shape = {shape_trk}")
    # logger.log(f"NeutralPFO Shape = {shape_neut_pfo}")
    # logger.log(f"ShotPFO Shape = {shape_shot_pfo}")
    # logger.log(f"ConvTracks Shape = {shape_conv_trk}")
    # logger.log(f"TauJets Shape = {shape_jet}")
    # logger.log(f"Labels Shape = {shape_label}")
    # logger.log(f"Weight Shape = {shape_weight}")

    types = (
            (tf.float32,
             tf.float32,
             tf.float32,
             tf.float32,
             tf.float32),
            tf.float32,
            tf.float32)


    shapes = (
              (tf.TensorShape([None, 14, 20]),
               tf.TensorShape([None, 22, 20]),
               tf.TensorShape([None, 6, 20]),
               tf.TensorShape([None, 10, 20]),
               tf.TensorShape([None, 9])),
              tf.TensorShape([None, 4]),
              tf.TensorShape([None])
            )

    train_dataset = tf.data.Dataset.from_generator(training_batch_generator, output_types=types, output_shapes=shapes)
    train_dataset = tf.data.Dataset.range(2).interleave(lambda _: train_dataset, num_parallel_calls=12,)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    train_dataset = train_dataset.repeat(100)

    validation_batch_generator = DataGenerator(validation_files_dictionary, variables_dictionary, nbatches=250, cuts=cuts,
                                               label="Validation Generator")
    val_dataset = tf.data.Dataset.from_generator(validation_batch_generator, output_types=types, output_shapes=shapes)
    val_dataset = tf.data.Dataset.range(2).interleave(lambda _: val_dataset, num_parallel_calls=12)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    #val_dataset  =val_dataset.repeat()

    #==================================================================================================================#
    # Initialize Model
    #==================================================================================================================#
    model = ModelDSNN(config_dict)

    # Configure callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss", min_delta=0.0001,
        patience=10, verbose=0, restore_best_weights=True)

    model_checkpoint = ParallelModelCheckpoint(model,
                                               path=os.path.join("data", 'weights-{epoch:02d}.h5'),
                                               monitor="val_loss", save_best_only=False, save_weights_only=True,
                                               verbose=0)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=3, min_lr=4e-6)
    cb = TimingCallback()

    callbacks = [early_stopping, model_checkpoint,  reduce_lr]

    model.summary()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Now initialise an iterator
    #train_iterator = iter(train_dataset)
    #val_iterator = iter(val_dataset)

    # Create two objects, x & y, from batch
    #train_x, train_y, train_weights =train_iterator.get_next()
    #val_x, val_y, val_weights = val_iterator.get_next()

    # ==================================================================================================================#
    # Train Model
    # ==================================================================================================================#
    history = model.fit(train_dataset, epochs=100, callbacks=callbacks,
                         validation_data=val_dataset, validation_freq=1, verbose=1, shuffle=True,
                        workers=6, steps_per_epoch=len(training_batch_generator), use_multiprocessing=True
                        )

    #max_queue_size=6, workers=6,  tf.data.AUTOTUNE,

if __name__ == "__main__":
    main()

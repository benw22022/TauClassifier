"""
Main Code Body
"""
import os
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'    # Acclerated Linear Algbra (XLA) Seems to actually make things slower
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                     # Disables GPU

from variables import variables_dictionary
from models import ModelDSNN
from DataGenerator import DataGenerator
from files import training_files, validation_files
from callbacks import ParallelModelCheckpoint, TimingCallback
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from utils import logger
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from config import config_dict, cuts, types, shapes
logger.set_log_level('INFO')


def main():
    logger.log("Beginning dataset preparation", 'INFO')

    # Initialize Generators
    training_batch_generator = DataGenerator(training_files, variables_dictionary, nbatches=2500, cuts=cuts,
                                             label="Training Generator")
    train_dataset = tf.data.Dataset.from_generator(training_batch_generator, output_types=types, output_shapes=shapes)
    #train_dataset = tf.data.Dataset.range(2).interleave(lambda _: train_dataset, num_parallel_calls=12,)
    # train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    # train_dataset.flat_map(lambda t1, t2, t3: tf.data.Dataset.from_tensors((t1, t2, t3)).repeat(2))


    train_dataset = train_dataset.repeat(100)

    validation_batch_generator = DataGenerator(validation_files, variables_dictionary, nbatches=2500, cuts=cuts,
                                               label="Validation Generator")
    #val_dataset = tf.data.Dataset.from_generator(validation_batch_generator, output_types=types, output_shapes=shapes)
    #val_dataset = tf.data.Dataset.range(2).interleave(lambda _: val_dataset, num_parallel_calls=12)
    #val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    #val_dataset  =val_dataset.repeat()

    # val_iterator = iter(val_dataset)
    #
    # # Create two objects, x & y, from batch
    # # train_x, train_y, train_weights =train_iterator.get_next()
    # val_x, val_y, val_weights = val_iterator.get_next()

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Initialize Model
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
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

    callbacks = [early_stopping, model_checkpoint, reduce_lr]

    model.summary()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
     Train Model
     """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    history = model.fit(training_batch_generator, epochs=100, callbacks=callbacks,
                        validation_data=validation_batch_generator, validation_freq=1, verbose=1, shuffle=True,
                        #workers=6, steps_per_epoch=10, use_multiprocessing=True, #(val_x, val_y, val_weights)
                        )
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Make Plots 
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("plots\\loss_history.svg")
    plt.show()
    # Save results
    with open('data\\loss_history.pkl', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(history["loss"], filehandle)

    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("plots\\accuracy_history.svg")
    plt.show()
    # Save results
    with open('data\\loss_history.pkl', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(history["accuracy"], filehandle)

    #max_queue_size=6, workers=6,  tf.data.AUTOTUNE,

if __name__ == "__main__":
    main()

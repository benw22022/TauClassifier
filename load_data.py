import tensorflow as tf
import os
import glob
from config.config import config_dict, get_cuts, models_dict
import time
import tqdm

def load_data(file_path):
    ds = tf.data.experimental.load(file_path.numpy().decode("utf-8"))
    return ds

def load_data_wrapper(file_path):
    return tf.py_function(load_data, [file_path], [tf.string])

def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]

def concat_datasets(datasets):
    dataset = datasets[0]
    for i in tqdm.tqdm(range(1, len(datasets))):
        dataset = dataset.concatenate(datasets[i])
    return dataset

if __name__ == "__main__":


    dataset_list = glob.glob("data/train_data/*.dat")

    time_start = time.time()
    datasets = [tf.data.experimental.load(file) for file in dataset_list]

    # data1, data2 = split_list(datasets)
    # data1 = concat_datasets(data1)
    # data2 = concat_datasets(data2)
    
    # datasets = [data1, data2]

    # dataset = dataset.interleave(datasets, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # dataset = tf.data.Dataset.from_tensor_slices(datasets).interleave(lambda x: x, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    dataset = concat_datasets(datasets)
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    model = models_dict["DSNN"](config_dict)

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3) # default lr = 1e-3
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=[tf.keras.metrics.CategoricalAccuracy()])

    history = model.fit(dataset, epochs=200)
    
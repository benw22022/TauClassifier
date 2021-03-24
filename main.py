"""
Main Code Body
"""
from variables import input_variables, variables_dictionary
from models import experimental_model
from DataGenerator import DataGenerator
from dataset_maker import *


if __name__ == "__main__":

    time_memory_logger = TimeMemoryMonitor()

    # Grab all root files in directory
    # TODO: Make config file for datasets
    data_dir = "E:\\NTuples"
    tree_name = ":tree"
    all_files = get_root_files(data_dir, tree_name)

    path = "E:\\NTuples\\TauID\\"
    file_dict = {"Gammatautau": [path+"Gammatautau_1.root"],
                 "JZ3": [path+"JZ3W_4.root"]}


    # Initialize Generators
    training_batch_generator = DataGenerator(file_dict, variables_dictionary, 100000)
    #testing_batch_generator = DataGenerator(X_test_idx, variables_dictionary)
    #validation_batch_generator = DataGenerator(X_val_idx, variables_dictionary)

    # Initialize Model
    shape_trk, shape_cls, shape_jet, _, _ = training_batch_generator.get_batch_shapes()

    print(f"Track shape = {shape_trk}   Cluster shape = {shape_cls}    Jet shape = {shape_jet}")

    model = experimental_model(shape_trk[1:], shape_cls[1:], shape_jet[1:])
    model.summary()
    model.compile(optimizer="adam", loss="binary_crossentropy",  metrics=["accuracy"])

    # Train Model
    history = model.fit(training_batch_generator, epochs=100, max_queue_size=4, use_multiprocessing=False, shuffle=True)


import uproot
import tracemalloc  # Monitor memory usage
from sklearn import preprocessing
import time
from variables import input_variables, variables
from dataloader import DataLoader
from models import experimental_model
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
from DataGenerator import DataGenerator
import awkward as ak
import pandas as pd
import gc
from sys import getsizeof



if __name__ == "__main__":

    # Start counting time and memory usage
    start_time = time.time()
    tracemalloc.start()



    #make_files(signal_files, input_variables, "Gammatautau", 1)
    arr = np.load("data\\180\\test_TauJetsAuxDyn.trk_nInnermostPixelHits_180.npz", allow_pickle=True)["arr_0"]
    print(arr)


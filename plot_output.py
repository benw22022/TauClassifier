from operator import imod
import os
import tqdm 
import glob
import yaml
import ray
import random
from sklearn.model_selection import train_test_split
from source.datawriter import RayDataWriter, DataWriter


if __name__ == "__main__":
    
    # Load yaml config file 
    with open("config/file_config.yaml", 'r') as stream:
        file_config = yaml.load(stream, Loader=yaml.FullLoader)
    
    # Grab testing files
    tau_files = glob.glob(file_config["TauFiles"])
    jet_files = glob.glob(file_config["FakeFiles"])
    _, tau_test_files = train_test_split(tau_files, test_size=file_config["TestSplit"], random_state=file_config["RandomSeed"])
    _, jet_test_files = train_test_split(jet_files, test_size=file_config["TestSplit"], random_state=file_config["RandomSeed"])
    
    
    os.system("rm -r control_plots/*")
    file = random.choice(tau_files)
    
    loader = DataWriter(file, "config/features.yaml")
    loader.make_control_plots()
    
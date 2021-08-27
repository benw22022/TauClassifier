"""
Files
______________________________________
Initializes the file handlers
TODO: Should probably work out how to initialize the file handlers using a YAML config file
TODO: We don't really want other people to have to edit .py files unless necessary
"""

from pathlib import Path
from scripts.utils import logger
from scripts.utils import FileHandler

#ntuple_dir = "C:\\Users\\benwi\\TauClassifier"
ntuple_dir = "/eos/user/b/bewilson/TauClassifier/new_NTuples"

gammatautau_files = FileHandler("Gammatautau", str(Path(f"{ntuple_dir}/*Gammatautau*/*.root")), class_label=1)
jz1_files = FileHandler("JZ1", str(Path(f"{ntuple_dir}/*JZ1*/*.root")), class_label=0)
jz2_files = FileHandler("JZ2", str(Path(f"{ntuple_dir}/*JZ2*/*.root")), class_label=0)
jz3_files = FileHandler("JZ3", str(Path(f"{ntuple_dir}/*JZ3*/*.root")), class_label=0)
jz4_files = FileHandler("JZ4", str(Path(f"{ntuple_dir}/*JZ4*/*.root")), class_label=0)
jz5_files = FileHandler("JZ5", str(Path(f"{ntuple_dir}/*JZ5*/*.root")), class_label=0)
jz6_files = FileHandler("JZ6", str(Path(f"{ntuple_dir}/*JZ6*/*.root")), class_label=0)
jz7_files = FileHandler("JZ7", str(Path(f"{ntuple_dir}/*JZ7*/*.root")), class_label=0)
jz8_files = FileHandler("JZ8", str(Path(f"{ntuple_dir}/*JZ8*/*.root")), class_label=0)
jz_files = FileHandler("JZ", str(Path(f"{ntuple_dir}/*JZ*/*.root")), class_label=0)

#gammatautau_files = FileHandler("Gammatautau", "/mnt/e/NTuples/TauClassifier/*Gammatautau*/*.root", class_label=1)
#jz1_files = FileHandler("JZ1",  "/mnt/e/NTuples/TauClassifier/*JZ1*/*.root", class_label=0)

training_files = [gammatautau_files[:-2], jz1_files[:-2], jz2_files[:-2], jz3_files[:-2], jz4_files[:-2], jz5_files[:-2],
                  jz6_files[:-2], jz7_files[:-2], jz8_files[:-2]]


#training_files = [gammatautau_files[0], jz1_files[0], jz2_files[0], jz3_files[0], jz4_files[0], jz5_files[0],
#                  jz6_files[0], jz7_files[0], jz8_files[0]]

validation_files = [gammatautau_files[-2:-1], jz1_files[-2:-1], jz2_files[-2:-1], jz3_files[-2:-1], jz4_files[-2:-1],
                    jz5_files[-2:-1], jz6_files[-2:-1], jz7_files[-2:-1], jz8_files[-2:-1]]

testing_files = [gammatautau_files[-1:], jz1_files[-1:], jz2_files[-1:], jz3_files[-1:], jz4_files[-1:], jz5_files[-1:],
                 jz6_files[-1:], jz7_files[-1:], jz8_files[-1:]]
#testing_files = [gammatautau_files[-1:], jz1_files[-1:]]


# Sanity checks to make sure we don't mix up datasets
for training_fh in training_files:

    for validation_fh in validation_files:
        for file in training_fh.file_list:
            if file in validation_fh.file_list:
                 logger.log(f"Training file {file} is also a part of the validation data set!", 'ERROR')

    for testing_fh in testing_files:
        for file in training_fh.file_list:
            if file in testing_fh.file_list:
                 logger.log(f"Training file {file} is also a part of the testing data set!", 'ERROR')

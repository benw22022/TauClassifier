"""
Just file definitions
Note: If no files are appearing check that the '/' in the glob statements is NOT a '\\' (This is a side effect from development in Windows)
"""

import glob as glob

ntuple_dir = "/eos/user/b/bewilson/TauClassifier/NTuples"
tree = "tree"

training_files_dictionary = {"Gammatautau": glob.glob(f"{ntuple_dir}/*Gammatautau*/*.root")[:-1],
                             "JZ1": glob.glob(f"{ntuple_dir}/*JZ1*/*.root")[:-1],
                             "JZ2": glob.glob(f"{ntuple_dir}/*JZ2*/*.root")[:-1],
                             "JZ3": glob.glob(f"{ntuple_dir}/*JZ3*/*.root")[:-1],
                             "JZ4": glob.glob(f"{ntuple_dir}/*JZ4*/*.root")[:-1],
                             "JZ5": glob.glob(f"{ntuple_dir}/*JZ5*/*.root")[:-1],
                             "JZ6": glob.glob(f"{ntuple_dir}/*JZ6*/*.root")[:-1],
                             "JZ7": glob.glob(f"{ntuple_dir}/*JZ7*/*.root")[:-1],
                             "JZ8": glob.glob(f"{ntuple_dir}/*JZ8*/*.root")[:-1],
                             }

validation_files_dictionary = {"Gammatautau": glob.glob(f"{ntuple_dir}/*Gammatautau*/*.root")[-1:],
                                "JZ1": glob.glob(f"{ntuple_dir}/*JZ1*/*.root")[-1:],
                                "JZ2": glob.glob(f"{ntuple_dir}/*JZ2*/*.root")[-1:],
                                "JZ3": glob.glob(f"{ntuple_dir}/*JZ3*/*.root")[-1:],
                                "JZ4": glob.glob(f"{ntuple_dir}/*JZ4*/*.root")[-1:],
                                "JZ5": glob.glob(f"{ntuple_dir}/*JZ5*/*.root")[-1:],
                                "JZ6": glob.glob(f"{ntuple_dir}/*JZ6*/*.root")[-1:],
                                "JZ7": glob.glob(f"{ntuple_dir}/*JZ7*/*.root")[-1:],
                                "JZ8": glob.glob(f"{ntuple_dir}/*JZ8*/*.root")[-1:],
                                }

print(training_files_dictionary)

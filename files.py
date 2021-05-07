"""
Just file definitions
TODO: This should probably have a YAML config file
"""

import glob as glob

ntuple_dir = "/eos/user/b/bewilson/TauClassifier/NTuples"
tree = "tree"

training_files_dictionary = {"Gammatautau": glob.glob(f"{ntuple_dir}\\*Gammatautau*\\*.root")[:-1],
                             "JZ1": glob.glob(f"{ntuple_dir}\\*JZ1*\\*.root")[:-2],
                             "JZ2": glob.glob(f"{ntuple_dir}\\*JZ2*\\*.root")[:-2],
                             "JZ3": glob.glob(f"{ntuple_dir}\\*JZ3*\\*.root")[:-2],
                             "JZ4": glob.glob(f"{ntuple_dir}\\*JZ4*\\*.root")[:-2],
                             "JZ5": glob.glob(f"{ntuple_dir}\\*JZ5*\\*.root")[:-2],
                             "JZ6": glob.glob(f"{ntuple_dir}\\*JZ6*\\*.root")[:-2],
                             "JZ7": glob.glob(f"{ntuple_dir}\\*JZ7*\\*.root")[:-2],
                             "JZ8": glob.glob(f"{ntuple_dir}\\*JZ8*\\*.root")[:-2],
                             }

validation_files_dictionary = {"Gammatautau": glob.glob(f"{ntuple_dir}\\*Gammatautau*\\*.root")[-1:],
                               "JZ1": glob.glob(f"{ntuple_dir}\\*JZ1*\\*.root")[-2:-1],
                               "JZ2": glob.glob(f"{ntuple_dir}\\*JZ2*\\*.root")[-2:-1],
                               "JZ3": glob.glob(f"{ntuple_dir}\\*JZ3*\\*.root")[-2:-1],
                               "JZ4": glob.glob(f"{ntuple_dir}\\*JZ4*\\*.root")[-2:-1],
                               "JZ5": glob.glob(f"{ntuple_dir}\\*JZ5*\\*.root")[-2:-1],
                               "JZ6": glob.glob(f"{ntuple_dir}\\*JZ6*\\*.root")[-2:-1],
                               "JZ7": glob.glob(f"{ntuple_dir}\\*JZ7*\\*.root")[-2:-1],
                               "JZ8": glob.glob(f"{ntuple_dir}\\*JZ8*\\*.root")[-2:-1],
                               }

testing_files_dictionary = {"Gammatautau": glob.glob(f"{ntuple_dir}\\*Gammatautau*\\*.root")[-1:],
                            "JZ1": glob.glob(f"{ntuple_dir}\\*JZ1*\\*.root")[-1:],
                            "JZ2": glob.glob(f"{ntuple_dir}\\*JZ2*\\*.root")[-1:],
                            "JZ3": glob.glob(f"{ntuple_dir}\\*JZ3*\\*.root")[-1:],
                            "JZ4": glob.glob(f"{ntuple_dir}\\*JZ4*\\*.root")[-1:],
                            "JZ5": glob.glob(f"{ntuple_dir}\\*JZ5*\\*.root")[-1:],
                            "JZ6": glob.glob(f"{ntuple_dir}\\*JZ6*\\*.root")[-1:],
                            "JZ7": glob.glob(f"{ntuple_dir}\\*JZ7*\\*.root")[-1:],
                            "JZ8": glob.glob(f"{ntuple_dir}\\*JZ8*\\*.root")[-1:],
                            }

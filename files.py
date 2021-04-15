"""
Just file definitions
"""

import glob as glob

ntuple_dir = "E:\\NTuples\\TauClassifier"
tree = "tree"

files_dictionary = {"Gammatautau": glob.glob(f"{ntuple_dir}\\*Gammatautau*\\*.root"),
                    "JZ1": glob.glob(f"{ntuple_dir}\\*JZ1*\\*.root"),
                    "JZ2": glob.glob(f"{ntuple_dir}\\*JZ2*\\*.root"),
                    "JZ3": glob.glob(f"{ntuple_dir}\\*JZ3*\\*.root"),
                    "JZ4": glob.glob(f"{ntuple_dir}\\*JZ4*\\*.root"),
                    "JZ5": glob.glob(f"{ntuple_dir}\\*JZ5*\\*.root"),
                    "JZ6": glob.glob(f"{ntuple_dir}\\*JZ6*\\*.root"),
                    "JZ7": glob.glob(f"{ntuple_dir}\\*JZ7*\\*.root"),
                    "JZ8": glob.glob(f"{ntuple_dir}\\*JZ8*\\*.root"),
                    }

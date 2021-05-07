"""
Files
______________________________________
Initializes the file handlers
TODO: Should probably work out how to initialize the file handlers using a YAML config file
TODO: We don't really want other people to have to edit .py files unless necessary
"""

from utils import FileHandler

ntuple_dir = "E:\\NTuples\\TauClassifier"

gammatautau_files = FileHandler("Gammatautau", f"{ntuple_dir}\\*Gammatautau*\\*.root", class_label=1)
jz1_files = FileHandler("JZ1", f"{ntuple_dir}\\*JZ1*\\*.root", class_label=0)
jz2_files = FileHandler("JZ2", f"{ntuple_dir}\\*JZ2*\\*.root", class_label=0)
jz3_files = FileHandler("JZ3", f"{ntuple_dir}\\*JZ3*\\*.root", class_label=0)
jz4_files = FileHandler("JZ4", f"{ntuple_dir}\\*JZ4*\\*.root", class_label=0)
jz5_files = FileHandler("JZ5", f"{ntuple_dir}\\*JZ5*\\*.root", class_label=0)
jz6_files = FileHandler("JZ6", f"{ntuple_dir}\\*JZ6*\\*.root", class_label=0)
jz7_files = FileHandler("JZ7", f"{ntuple_dir}\\*JZ7*\\*.root", class_label=0)
jz8_files = FileHandler("JZ8", f"{ntuple_dir}\\*JZ8*\\*.root", class_label=0)


training_files = [gammatautau_files[:-2], jz1_files[:-2], jz2_files[:-2], jz3_files[:-2], jz4_files[:-2], jz5_files[:-2],
                  jz6_files[:-2], jz7_files[:-2], jz8_files[:-2]]

validation_files = [gammatautau_files[-2:-1], jz1_files[-2:-1], jz2_files[-2:-1], jz3_files[-2:-1], jz4_files[-2:-1],
                    jz5_files[-2:-1], jz6_files[-2:-1], jz7_files[-2:-1], jz8_files[-2:-1]]

testing_files = [gammatautau_files[-1:], jz1_files[-1:], jz2_files[-1:], jz3_files[-1:], jz4_files[-1:], jz5_files[-1:],
                 jz6_files[-1:], jz7_files[-1:], jz8_files[-1:]]


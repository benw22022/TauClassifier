"""
Write file configs
____________________________________________________________________________
Script to automatically write train, test and validation file configs
Easier than writing them out by hand
"""

from sklearn.model_selection import train_test_split
import glob
import os

def write_file_config(filename, file_dict, ntuple_dir, mode):

    with open(filename, 'w') as file:
        file.write("---\n\n")
        file.write(f"Dataset: {mode}\n\n")
        file.write(f"NTupleDir: {ntuple_dir}\n\n")
        for file_type, file_list in file_dict.items():
            file.write(f"{file_type}:\n")
            is_tau = True
            if "JZ" in file_type:
                is_tau = False
            file.write(f"\t- is_tau: {is_tau}\n")
            file.write(f"\t- Files:\n")
            for f in file_list:
                file.write(f"\t\t- {os.path.basename(f)}\n")
            file.write("\n")

def get_train_test_val(files, test_split=0.2, val_split=0.2):

    train_files, test_files = train_test_split(files, test_size=test_split, random_state=42)
    train_files, val_files= train_test_split(train_files, test_size=val_split, random_state=42)

    return train_files, test_files, val_files



if __name__ == "__main__":

    tau_files = get_train_test_val(glob.glob("../../split_NTuples/Gammatautau/*.root"))
    
    JZ1_files = get_train_test_val(glob.glob("../../split_NTuples/JZ1/*.root"))
    JZ2_files = get_train_test_val(glob.glob("../../split_NTuples/JZ2/*.root"))
    JZ3_files = get_train_test_val(glob.glob("../../split_NTuples/JZ3/*.root"))
    JZ4_files = get_train_test_val(glob.glob("../../split_NTuples/JZ4/*.root"))
    JZ5_files = get_train_test_val(glob.glob("../../split_NTuples/JZ5/*.root"))
    JZ6_files = get_train_test_val(glob.glob("../../split_NTuples/JZ6/*.root"))
    JZ7_files = get_train_test_val(glob.glob("../../split_NTuples/JZ7/*.root"))
    JZ8_files = get_train_test_val(glob.glob("../../split_NTuples/JZ8/*.root"))


    train_file_dict = {"Gammatautau": tau_files[0],
                               "JZ1": JZ1_files[0],
                               "JZ2": JZ2_files[0],
                               "JZ3": JZ3_files[0],
                               "JZ4": JZ4_files[0],
                               "JZ5": JZ5_files[0],
                               "JZ6": JZ6_files[0],
                               "JZ7": JZ7_files[0],
                               "JZ8": JZ8_files[0]}

    test_file_dict = {"Gammatautau": tau_files[1],
                               "JZ1": JZ1_files[1],
                               "JZ2": JZ2_files[1],
                               "JZ3": JZ3_files[1],
                               "JZ4": JZ4_files[1],
                               "JZ5": JZ5_files[1],
                               "JZ6": JZ6_files[1],
                               "JZ7": JZ7_files[1],
                               "JZ8": JZ8_files[1]}

    val_file_dict = {"Gammatautau": tau_files[2],
                               "JZ1": JZ1_files[2],
                               "JZ2": JZ2_files[2],
                               "JZ3": JZ3_files[2],
                               "JZ4": JZ4_files[2],
                               "JZ5": JZ5_files[2],
                               "JZ6": JZ6_files[2],
                               "JZ7": JZ7_files[2],
                               "JZ8": JZ8_files[2]}
    
    write_file_config("../config/train_files_config.yaml", train_file_dict, "../../split_NTuples", "Train")
    write_file_config("../config/test_files_config.yaml", test_file_dict, "../../split_NTuples", "Test")
    write_file_config("../config/validation_files_config.yaml", val_file_dict, "../../split_NTuples", "Validation")
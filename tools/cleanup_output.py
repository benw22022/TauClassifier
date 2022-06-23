"""
Clean Up
_____________________________________________________________
Clean up outputs directory by deleting files
"""

import os
import glob
import shutil


def main():

    output_dirs = glob.glob(os.path.join("outputs", "*_output"))
    
    for folder in output_dirs:
        if 'train' not in folder:
            print(f"Deleting: {folder}")
            shutil.rmtree(folder)
        else:
            training_dirs = glob.glob(os.path.join(folder , '*'))
            for training_run in training_dirs:
                weights = glob.glob(os.path.join(training_run, "network_weights", "*"))
                if len(weights) == 0:
                    print(f"Deleting: {training_run}")                
                    shutil.rmtree(training_run)
    


if __name__ == "__main__":
    main()
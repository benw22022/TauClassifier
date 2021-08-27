#!\bin\python3
"""
TauClassifier
_______________________________________________________________
Main steering script for running the Tau Classifier
"""

import os
import argparse
import getpass
from datetime import datetime
from shutil import copyfile
from run.train import train
from run.test import test
from scripts.utils import logger


def run_training_on_batch_system(prong=None, log_level=None):
    
    #  Check that user has changed the submit file to their email if they plan on using the batch system
    if getpass.getuser() != "bewilson":
        with open("CommonFiles/htc_generation.submit", "r") as submit_file:
            for line in submit_file:
                assert "notify_user" not in line and "benjamin.james.wilson@cern.ch" not in line,\
                    "In batch/htc_generation.submit please change the notify_user field to your email!"

    # Make a new directory to run in
    current_dir = os.getcwd()
    now = datetime.now()
    new_dir = "../training_" + now.strftime("%Y-%m-%d_%H.%M.%S")
    os.system(f"mkdir {new_dir}")
    logger.log(f"Created new directory to run in {new_dir} ")
    os.system(f"cp batch/htc_training.submit {new_dir}")

    # Write a script that will run on the batch system
    script = f"""
#!/bin/bash 
# Activate the conda enviroment
eval "$(conda shell.bash hook)"
conda activate tauid

# Copy files to batch
rsync -av --progress {current_dir} {new_dir} --exclude .git* --exclude .idea --exclude *__pycache__

# Move to folder
cd {new_dir}/TauClassifier

# Run the training 
python3 tauclassifier.py train -prong={prong} -log_level={log_level} | tee training.log
"""
    with open(f"{new_dir}/train_on_batch.sh", 'w') as file:
        file.write(script)        

    # Move to new directory and run 
    os.chdir(new_dir)
    # os.system("condor_submit batch/htc_training.submit -batch-name TauClassifierTraining")
    os.system("chmod +x train_on_batch.sh")
    os.system("./train_on_batch.sh")


def none_or_int(value):
    if value == 'None':
        return None
    return value


def main():

    # Available options
    mode_list = ["train", "test"]
    prong_list = [1, 3, None]
    loglevels = ['ERROR', 'WARNING', 'INFO', 'DEBUG', 'HELPME ']

    # Get user arguements
    parser = argparse.ArgumentParser()
    parser.add_argument("run_mode", help="Either 'train' or 'test'", type=str, choices=mode_list) 
    parser.add_argument("-prong", help="Number of prongs - if 'None' will use 1+3prongs", type=none_or_int, choices=prong_list, default=None)
    parser.add_argument("-weights", help="File path to network weights to test", type=str, default='')
    parser.add_argument("-log_level", help="Sets log level", 
    type=str, default='INFO')
    parser.add_argument("-condor", help='Run on ht condor batch system', type=bool, default=False)
    args = parser.parse_args()

    # If training
    if args.run_mode == 'train':

        # If training on ht condor batch system
        if args.condor:
            run_training_on_batch_system(prong=args.prong, log_level=args.log_level)
            return 0

        # If training on local machine
        train(prong=args.prong, log_level=args.log_level)

    # If testing
    if args.run_mode == 'test':
        test()


if __name__ == "__main__":
    main()
#!/bin/python3
"""
TauClassifier
_______________________________________________________________
Main steering script for running the Tau Classifier
Example usage:
python3 tauclassifier.py train -condor=True   # Runs training on batch system
"""

import os
import argparse
import getpass
from pathlib import Path
from datetime import datetime
from shutil import copyfile
from run.train import train
from run.evaluate import evaluate
from run.permutation_rank import permutation_rank
from run.test import test
from scripts.utils import logger
from config.config import models_dict
import scratch
import faulthandler



def run_training_on_batch_system(prong=None, log_level=None, model='DSNN', tf_log_level='2'):
    
    #  Check that user has changed the submit file to their email if they plan on using the batch system
    if getpass.getuser() != "bewilson":
        with open("batch/htc_training.submit", "r") as submit_file:
            for line in submit_file:
                assert "benjamin.james.wilson@cern.ch" not in line, "In batch/htc_generation.submit please change the notify_user field to your email!"

    # Make a new directory to run in and copy code into it
    current_dir = os.getcwd()
    now = datetime.now()
    parent_dir = Path(current_dir).parent.absolute()
    new_dir = os.path.join(parent_dir, "training_" + now.strftime("%Y-%m-%d_%H.%M.%S"))
    os.mkdir(new_dir)
    logger.log(f"Created new directory to run in {new_dir} ")
    os.system(f"cp batch/htc_training.submit {new_dir}")
    os.system("rsync -ar --progress --exclude=.git --exclude=.idea --exclude=*pycache* {} {}".format(current_dir, new_dir))

    # Write a script that will run on the batch system (There is probably a easier way to do this but I can't figure it out)
    script = f"""
#!/bin/bash 
# Activate the conda enviroment
eval "$(conda shell.bash hook)"
conda activate tauid
echo "Conda Enviroment: " $CONDA_DEFAULT_ENV

# Move to folder
cd {new_dir}/TauClassifier

# Run the training 
python3 tauclassifier.py train -prong={prong} -log_level={log_level} -model={model} -tf_log_levl={tf_log_level} | tee training.log
"""
    with open(f"{new_dir}/train_on_batch.sh", 'w') as file:
        file.write(script)        

    # Move to new directory and run 
    os.chdir(new_dir)
    os.system("condor_submit -batch-name TauClassifierTraining htc_training.submit ")
    return 0

def none_or_int(value):
    if value == 'None':
        return None
    return value


def main():

    # Available options

    # 'train' - train model | 'evaluate' =  make npz files of predictions for test data | 'plot' - make performance plots
    # 'scratch' - run a standalone testing script. We want to be able to run it from here so imports work properly
    mode_list = ["train", "evaluate", "plot", "rank", "scratch"]  

    # Sets mode: 1 - (p10n, 1p1n, 1pxn, jets) | 3 - (3p0n, 3pxn, jets) | None - (p10n, 1p1n, 1pxn, 3p0n, 3pxn, jets)
    prong_list = [1, 3, None]                                           

    model_list = list(models_dict.keys())                                   # List of available models
    log_levels = ['ERROR', 'WARNING', 'INFO', 'DEBUG', 'HELPME ']      # Custom logging levels for program
    tf_log_levels = ['0', '1', '2', '3']                               # TF logging levels 

    # Get user arguements
    parser = argparse.ArgumentParser()
    parser.add_argument("run_mode", help="Either 'train' or 'evaluate' or 'plot'", type=str, choices=mode_list) 
    parser.add_argument("-prong", help="Number of prongs - if 'None' will use 1+3prongs", type=none_or_int, choices=prong_list, default=None)
    parser.add_argument("-weights", help="File path to network weights to test", type=str, default='')
    parser.add_argument("-model", help="Model to use: models are defined in model/models.py and registered in config/config.py in the models dictionary", choices=model_list, 
    type=str, default="DSNN")
    parser.add_argument("-ncores", help="number of CPU cores to use when evaluating network predictions", type=int, default=8)
    parser.add_argument("-log_level", help="Sets log level", type=str, default='INFO', choices=log_levels)
    parser.add_argument("-tf_log_level", help="Set Tensorflow logging level", type=str, default='2')
    parser.add_argument("-function", help="Scratch function to run")
    parser.add_argument("-condor", help='Run on ht condor batch system', type=bool, default=False)
    parser.add_argument("-load", help="Load last saved network predictions", type=bool, default=False)
    args = parser.parse_args()

    # If training
    if args.run_mode == 'train':

        # If training on ht condor batch system
        if args.condor:
            return run_training_on_batch_system(prong=args.prong, model=args.model, log_level=args.log_level, tf_log_level=args.tf_log_level)

        # If training on local machine
        # return train(prong=args.prong, model=args.model, log_level=args.log_level, tf_log_level=args.tf_log_level)

        with open("fault_handler.log", "w") as fobj:
            faulthandler.enable(fobj)

            return train(args)
            

    # If testing
    if args.run_mode == 'evaluate':
        return evaluate(args.weights, ncores=args.ncores)
    
    if args.run_mode == 'rank':
        return permutation_rank(args)

    # Make performance plots
    if args.run_mode == 'plot':
        return test()

    # *Super* hacky way of running little standalone testing scripts 
    if args.run_mode == 'scratch':
        return getattr(globals()[scratch], args.function)()   

        


if __name__ == "__main__":
    main()

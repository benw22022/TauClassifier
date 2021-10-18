#!/bin/python3
"""
TauClassifier
_______________________________________________________________
Main steering script for running the Tau Classifier
Example usage:
python3 tauclassifier.py train 
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
from run.testMK2 import test
from scripts.utils import logger, get_best_weights, none_or_int, run_training_on_batch_system
from config.config import models_dict
from scripts.utils import logger
import scratch
import faulthandler
import glob

def main():

    # Available options

    # 'train' - train model | 'evaluate' =  make npz files of predictions for test data | 'plot' - make performance plots
    # 'scratch' - run a standalone testing script. We want to be able to run it from here so imports work properly
    mode_list = ["train", "evaluate", "plot", "rank", "scratch", "scan"]  

    # Prong options: 1 - (p10n, 1p1n, 1pxn, jets) | 3 - (3p0n, 3pxn, jets) | None - (p10n, 1p1n, 1pxn, 3p0n, 3pxn, jets)
    prong_list = [1, 3, None]                                           

    model_list = list(models_dict.keys())                              # List of available models
    log_levels = ['ERROR', 'WARNING', 'INFO', 'DEBUG', 'HELPME ']      # Custom logging levels for program
    tf_log_levels = ['0', '1', '2', '3']                               # TF logging levels 

    # Last weights file produced by training 
    best_weights = get_best_weights()

    # Get user arguements
    parser = argparse.ArgumentParser()
    parser.add_argument("run_mode", help="Either 'train' or 'evaluate' or 'plot'", type=str, choices=mode_list) 
    parser.add_argument("-prong", help="Number of prongs - if 'None' will use 1+3prongs", type=none_or_int, choices=prong_list, default=None)
    parser.add_argument("-weights", help="File path to network weights to test", type=str, default=best_weights)
    parser.add_argument("-model", help="Model to use: models are defined in model/models.py and registered in config/config.py in the models dictionary", choices=model_list, 
    type=str, default="DSNN")
    parser.add_argument("-lr", help="Learning rate of Adam optimiser", type=float, default=1e-3)
    parser.add_argument("-lr_range", help="Learning rate array to scan through usage: -lr_range <start> <stop> <step>", type=float, nargs=3, default=[1e-4, 1e-2, 10])
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
        return train(args)
            

    # If testing
    if args.run_mode == 'evaluate':
        return evaluate(args.weights, ncores=args.ncores)
    
    # If permutation ranking
    if args.run_mode == 'rank':
        return permutation_rank(args)

    # Make performance plots
    if args.run_mode == 'plot':
        return test()
    
    # Scan through learning rates
    if args.run_mode == 'scan':
        try:
            int(args.lr_range[2])
        except ValueError:
            logger.log("Learning rate step size must be an integer!", "ERROR")
            return 1

    # *Super* hacky way of running little standalone testing scripts 
    if args.run_mode == 'scratch':
        return getattr(globals()[scratch], args.function)()   

        


if __name__ == "__main__":
    main()

"""
Evaluate.py
___________________________________________________________________
Compute predictions using a weights file
Writes out y_pred array for each NTuple into a .npz
"""

import ray
from config.config import get_cuts, config_dict
from scripts.utils import logger
from config.variables import variables_dictionary
from config.files import gammatautau_files, jz_files, testing_files, ntuple_dir, all_files
from scripts.DataLoader import DataLoader
from scripts.preprocessing import Reweighter

def split_list(alist, wanted_parts=1):
    """
    Splits a list into list of smaller lists
    :param alist: A list to split up
    :param wanted_parts: Number of parts to split alist into
    :returns: A split up list
    """
    length = len(alist)
    return [alist[i * length // wanted_parts: (i + 1) * length // wanted_parts]
			for i in range(wanted_parts)]

def evaluate(args):
    """
    Evaluates the network output for each NTuple and writes them to an npz file
    :param args: An argparse.Namespace object. Args that must be parsed:
                 -weights: A path to the network weights to evaluate
                 -ncores: Number of files to process in parallel
    """
    # Initialize Ray
    ray.init()

    # Load model
    model_config = config_dict
    model_weights = args.weights
    reweighter = Reweighter(ntuple_dir, prong=args.prong)
    assert model_weights != "", logger.log("\nYou must specify a path to the model weights", 'ERROR')

    # Get files
    files = all_files.file_list
    nbatches = 250
    
    # Split files into groups to speed things up, will process args.ncores files in parallel
    if args.ncores > len(files):
        args.ncores = len(files)
    if args.ncores < 0:
        args.ncores = 1
    files = split_list(files, len(files)//args.ncores)   
    
    # Make DataLoaders
    for file_chunk in files:
        dataloaders = []
        for file in file_chunk:
                dl = DataLoader.remote(file, [file], 1, nbatches, variables_dictionary, cuts=get_cuts(args.prong)[file.label], 
                                    reweighter=reweighter, no_gpu=True)
                dataloaders.append(dl)
        # Save predictions for each file in parallel
        ray.get([dl.predict.remote(args.model, model_config, model_weights, save_predictions=True) for dl in dataloaders])
        for dl in dataloaders:
            ray.kill(dl)
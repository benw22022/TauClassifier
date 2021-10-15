"""
Variable Permutation Ranking
_________________________________________________________________________
Ranks variable importance by shuffling a the array of a variable within a 
batch and computes the network loss. The larger the difference in loss
compared to the base line the more important that variable is
"""

import os
import pandas as pd
from config.files import testing_files, ntuple_dir
from scripts.DataGenerator import DataGenerator
from scripts.utils import logger
from config.config import config_dict, get_cuts, models_dict
from scripts.preprocessing import Reweighter
from config.variables import variables_dictionary



def permutation_rank(args):

    # Check that weights were given
    if args.weights == "":
        logger.log("You must specify a weights file!", 'ERROR')
        return 1
    if not os.path.isfile(args.weights):
        logger.log(f"Could not open weights file: {args.weights}", 'ERROR')
        return 1

    # Initialize objects
    reweighter = Reweighter(ntuple_dir, prong=args.prong)
    cuts = get_cuts(args.prong)

    testing_batch_generator = DataGenerator(testing_files, variables_dictionary, nbatches=50, cuts=cuts,
                                             reweighter=reweighter, prong=args.prong, label="Ranking Generator", 
                                             no_gpu=True)

    _, _, _, baseline_loss, baseline_acc = testing_batch_generator.predict(args.model, config_dict, args.weights)

    logger.log(f"Baseline: Loss = {baseline_loss}   Accuracy = {baseline_acc}")

    # Store results in this dict
    results_dict = {"Variable": [], "Loss": [], "Accuracy": []}

    # Begin ranking    
    logger.log("Begining permutation variable ranking...")

    for variable in variables_dictionary["TauJets"]:
        _, _, _, loss, acc = testing_batch_generator.predict(args.model, config_dict, args.weights, shuffle_var=variable)
        delta_loss = loss - baseline_loss
        delta_acc = acc - baseline_acc
        results_dict["Variable"].append(variable)        
        results_dict["Loss"].append(delta_loss)
        results_dict["Accuracy"].append(delta_acc)
        logger.log(f"{variable} -- loss difference = {delta_loss}   accuracy difference = {delta_acc}")

    for variable in variables_dictionary["TauTracks"]:
        _, _, _, loss, acc = testing_batch_generator.predict(args.model, config_dict, args.weights, shuffle_var=variable)
        delta_loss = loss - baseline_loss
        delta_acc = acc - baseline_acc
        results_dict["Variable"].append(variable)
        results_dict["Loss"].append(delta_loss)
        results_dict["Accuracy"].append(delta_acc)
        logger.log(f"{variable} -- loss difference = {delta_loss}   accuracy difference = {delta_acc}")

    for variable in variables_dictionary["ConvTrack"]:
        _, _, _, loss, acc = testing_batch_generator.predict(args.model, config_dict, args.weights, shuffle_var=variable)
        delta_loss = loss - baseline_loss 
        delta_acc = acc - baseline_acc
        results_dict["Variable"].append(variable)
        results_dict["Loss"].append(delta_loss)
        results_dict["Accuracy"].append(delta_acc)
        logger.log(f"{variable} -- loss difference = {delta_loss}   accuracy difference = {delta_acc}")

    for variable in variables_dictionary["ShotPFO"]:
        _, _, _, loss, acc = testing_batch_generator.predict(args.model, config_dict, args.weights, shuffle_var=variable)
        delta_loss = loss - baseline_loss 
        delta_acc = acc - baseline_acc
        results_dict["Variable"].append(variable)
        results_dict["Loss"].append(delta_loss)
        results_dict["Accuracy"].append(delta_acc)
        logger.log(f"{variable} -- loss difference = {delta_loss}   accuracy difference = {delta_acc}")

    for variable in variables_dictionary["NeutralPFO"]:
        _, _, _, loss, acc = testing_batch_generator.predict(args.model, config_dict, args.weights, shuffle_var=variable)
        delta_loss = loss - baseline_loss 
        delta_acc = acc - baseline_acc
        results_dict["Variable"].append(variable)
        results_dict["Loss"].append(delta_loss)
        results_dict["Accuracy"].append(delta_acc)
        logger.log(f"{variable} -- loss difference = {delta_loss}   accuracy difference = {delta_acc}")

    results_df = pd.DataFrame(data=results_dict)

    results_df = results_df.sort_values(by="Loss")
    print("\n\n\n***********************************************************************")
    print(results_df)

    results_df.to_csv("Permutation_Ranking.csv")

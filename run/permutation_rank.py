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
    """
    Perform permutation ranking by shuffling the data belonging to one variable at a time and computing the change in loss
    The larger the change in loss the more important that variable is to the model
    :param args: Arguements from tauclassifier.py - easier to just parse this rather than the indiviual arguments
    """

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
                                             reweighter=reweighter, prong=args.prong, label="Ranking Generator")

    testing_batch_generator.load_model(args.model, config_dict, args.weights)
    _, _, _, baseline_loss, baseline_acc = testing_batch_generator.predict(make_confusion_matrix=True)

    logger.log(f"Baseline: Loss = {baseline_loss}   Accuracy = {baseline_acc}")

    # Store results in this dict
    results_dict = {"Variable": [], "Loss": [], "Accuracy": []}

    # Begin ranking    
    logger.log("Begining permutation variable ranking...")

    # for variable in variables_dictionary["TauJets"]:
    #     _, _, _, loss, acc = testing_batch_generator.predict(shuffle_var=variable, make_confusion_matrix=True)
    #     delta_loss = loss -  baseline_loss # If delta loss is positive then variable is important to the NN
    #     delta_acc = acc - baseline_acc
    #     results_dict["Variable"].append(variable)        
    #     results_dict["Loss"].append(delta_loss)
    #     results_dict["Accuracy"].append(delta_acc)
    #     logger.log(f"{variable} -- loss difference = {delta_loss}   accuracy difference = {delta_acc}")

    # for variable in variables_dictionary["TauTracks"]:
    #     _, _, _, loss, acc = testing_batch_generator.predict(shuffle_var=variable, make_confusion_matrix=True)
    #     delta_loss = loss - baseline_loss
    #     delta_acc = acc - baseline_acc
    #     results_dict["Variable"].append(variable)
    #     results_dict["Loss"].append(delta_loss)
    #     results_dict["Accuracy"].append(delta_acc)
    #     logger.log(f"{variable} -- loss difference = {delta_loss}   accuracy difference = {delta_acc}")

    # for variable in variables_dictionary["ConvTrack"]:
    #     _, _, _, loss, acc = testing_batch_generator.predict(shuffle_var=variable, make_confusion_matrix=True)
    #     delta_loss = loss - baseline_loss 
    #     delta_acc = acc - baseline_acc
    #     results_dict["Variable"].append(variable)
    #     results_dict["Loss"].append(delta_loss)
    #     results_dict["Accuracy"].append(delta_acc)
    #     logger.log(f"{variable} -- loss difference = {delta_loss}   accuracy difference = {delta_acc}")

    # for variable in variables_dictionary["ShotPFO"]:
    #     _, _, _, loss, acc = testing_batch_generator.predict(shuffle_var=variable, make_confusion_matrix=True)
    #     delta_loss = loss - baseline_loss 
    #     delta_acc = acc - baseline_acc
    #     results_dict["Variable"].append(variable)
    #     results_dict["Loss"].append(delta_loss)
    #     results_dict["Accuracy"].append(delta_acc)
    #     logger.log(f"{variable} -- loss difference = {delta_loss}   accuracy difference = {delta_acc}")

    # for variable in variables_dictionary["NeutralPFO"]:
    #     _, _, _, loss, acc = testing_batch_generator.predict(shuffle_var=variable, make_confusion_matrix=True)
    #     delta_loss = loss - baseline_loss 
    #     delta_acc = acc - baseline_acc
    #     results_dict["Variable"].append(variable)
    #     results_dict["Loss"].append(delta_loss)
    #     results_dict["Accuracy"].append(delta_acc)
    #     logger.log(f"{variable} -- loss difference = {delta_loss}   accuracy difference = {delta_acc}")

    for idx, variable in enumerate(variables_dictionary["TauJets"]):
        _, _, _, loss, acc = testing_batch_generator.predict(shuffle_var=("TauJets", idx), make_confusion_matrix=True)
        delta_loss = loss -  baseline_loss # If delta loss is positive then variable is important to the NN
        delta_acc = acc - baseline_acc
        results_dict["Variable"].append(variable)        
        results_dict["Loss"].append(delta_loss)
        results_dict["Accuracy"].append(delta_acc)
        logger.log(f"{variable} -- loss difference = {delta_loss}   accuracy difference = {delta_acc}")

    for idx, variable in enumerate(variables_dictionary["TauTracks"]):
        _, _, _, loss, acc = testing_batch_generator.predict(shuffle_var=("TauTracks", idx), make_confusion_matrix=True)
        delta_loss = loss - baseline_loss
        delta_acc = acc - baseline_acc
        results_dict["Variable"].append(variable)
        results_dict["Loss"].append(delta_loss)
        results_dict["Accuracy"].append(delta_acc)
        logger.log(f"{variable} -- loss difference = {delta_loss}   accuracy difference = {delta_acc}")

    for idx, variable in enumerate(variables_dictionary["ConvTrack"]):
        _, _, _, loss, acc = testing_batch_generator.predict(shuffle_var=("ConvTrack", idx), make_confusion_matrix=True)
        delta_loss = loss - baseline_loss 
        delta_acc = acc - baseline_acc
        results_dict["Variable"].append(variable)
        results_dict["Loss"].append(delta_loss)
        results_dict["Accuracy"].append(delta_acc)
        logger.log(f"{variable} -- loss difference = {delta_loss}   accuracy difference = {delta_acc}")

    for idx, variable in enumerate(variables_dictionary["ShotPFO"]):
        _, _, _, loss, acc = testing_batch_generator.predict(shuffle_var=("ShotPFO", idx), make_confusion_matrix=True)
        delta_loss = loss - baseline_loss 
        delta_acc = acc - baseline_acc
        results_dict["Variable"].append(variable)
        results_dict["Loss"].append(delta_loss)
        results_dict["Accuracy"].append(delta_acc)
        logger.log(f"{variable} -- loss difference = {delta_loss}   accuracy difference = {delta_acc}")

    for idx, variable in enumerate(variables_dictionary["NeutralPFO"]):
        _, _, _, loss, acc = testing_batch_generator.predict(shuffle_var=("NeutralPFO", idx), make_confusion_matrix=True)
        delta_loss = loss - baseline_loss 
        delta_acc = acc - baseline_acc
        results_dict["Variable"].append(variable)
        results_dict["Loss"].append(delta_loss)
        results_dict["Accuracy"].append(delta_acc)
        logger.log(f"{variable} -- loss difference = {delta_loss}   accuracy difference = {delta_acc}")


    # Create pandas DataFrame of the results and sort by loss difference
    results_df = pd.DataFrame(data=results_dict)
    results_df = results_df.sort_values(by="Loss")
    print("\n\n\n***********************************************************************")
    print(results_df)

    # Save results to file
    results_df.to_csv("Permutation_Ranking.csv")

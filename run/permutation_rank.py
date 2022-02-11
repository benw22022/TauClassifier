"""
Variable Permutation Ranking
_________________________________________________________________________
Ranks variable importance by shuffling a the array of a variable within a 
batch and computes the network loss. The larger the difference in loss
compared to the base line the more important that variable is
"""

import ray
import pandas as pd
from config.files import testing_files, ntuple_dir
from scripts.DataGenerator import DataGenerator
from scripts.utils import logger
from config.config import config_dict, get_cuts, models_dict
from scripts.preprocessing import Reweighter
from config.variables import variable_handler
from run.test import test


class Ranker:

    def __init__(self, args, var_handler):
        # Initialize objects
        reweighter = Reweighter(ntuple_dir, prong=args.prong)
        cuts = get_cuts(args.prong)
        self.var_handler = var_handler
        self.batch_generator = DataGenerator(testing_files, self.var_handler, nbatches=50, cuts=cuts,
                                             reweighter=reweighter, prong=args.prong, label="Ranking Generator")

        self.batch_generator.load_model(args.model, config_dict, args.weights)
        _, _, _, self.baseline_loss, self.baseline_acc = self.batch_generator.predict(make_confusion_matrix=True)

        logger.log(f"Baseline: Loss = {self.baseline_loss}   Accuracy = {self.baseline_acc}")

        # Store results in this dict
        self.results_dict = {"Variable": [], "Loss": [], "Accuracy": []}

        # Begin ranking    
        logger.log("Begining permutation variable ranking...")


    def rank(self, var_type):
        for idx, variable in enumerate(self.var_handler.get(var_type)):
            _, _, _, loss, acc = self.batch_generator.predict(shuffle_var=(var_type, idx), make_confusion_matrix=True)
            delta_loss = loss -  self.baseline_loss # If delta loss is positive then variable is important to the NN
            delta_acc = acc - self.baseline_acc
            self.results_dict["Variable"].append(variable.name)        
            self.results_dict["Loss"].append(delta_loss)
            self.results_dict["Accuracy"].append(delta_acc)
            logger.log(f"{variable.name:<} -- loss difference = {delta_loss:<}   accuracy difference = {delta_acc:<}")
        
    def finish(self, saveas="Permutation_Ranking.csv"):
        # Create pandas DataFrame of the results and sort by loss difference
        results_df = pd.DataFrame(data=self.results_dict)
        results_df = results_df.sort_values(by="Loss")
        # Save results to file
        results_df.to_csv(saveas)


def permutation_rank(args):
    """
    Perform permutation ranking by shuffling the data belonging to one variable at a time and computing the change in loss
    The larger the change in loss the more important that variable is to the model
    :param args: Arguements from tauclassifier.py - easier to just parse this rather than the indiviual arguments
    """
    
    test(args, shuffle_index=(0, 1))



    # ray.init()

    # variable_ranker = Ranker(args, variable_handler)
    # variable_ranker.rank("TauJets")
    # variable_ranker.rank("TauTracks")
    # variable_ranker.rank("NeutralPFO")
    # variable_ranker.rank("ShotPFO")
    # variable_ranker.rank("ConvTrack")
    # variable_ranker.finish()

    # ray.shutdown()
    
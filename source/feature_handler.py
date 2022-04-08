
from typing import List
from dataclasses import dataclass
import numpy as np
import uproot

@dataclass
class Feature:
    # Helper class to handle input features
    name: str
    mean: int = None
    std: int = None

class FeatureHandler:
    """
    Class to handle training features and other input variables
    """
    def __init__(self) -> None:
        self.branches = {}
        self.max_branch_items = {}
        
    def add_branch(self, branch_name: str, features: List[Feature], max_objects=1) -> None:
        # Add another key, value pair to branches dictionary
        self.branches[branch_name] = features
        self.max_branch_items[branch_name] = max_objects

    def __getitem__(self, key: str):
        # Returns a list of strings containing the names of features in this branch 
        return [f.name for f in self.branches[key]]
    
    def as_list(self) -> List[str]:
        # Return a list of all input variables in all branches 
        # - useful for filter_name option in uproot
        names = []
        for value in self.branches.values():
            if isinstance(value, list):
                for f in value:
                    names.append(f.name)
            else:
                names.append(value.name)
        return list(set(names))

    def get_stats(self, branch_name: str, feature_name: str):
        # Get mean and standard devation for a feature belonging to a specific branch
        for f in self.branches[branch_name]:
            if f.name == feature_name:
                return f.mean, f.std
    
    def max_objects(self, branch_name):
        return self.max_branch_items[branch_name]

    
    def get_pt_reweight(self, pt):
        
        return self.reweight_hist_values[np.digitize(pt, self.reweight_hist_dges)]

    def load_reweight_hist(self, filename: str, histname: str="ReWeightHist"):

        file = uproot.open(filename)
        self.reweight_hist_edges = file[histname].axis().edges()
        self.reweight_hist_values = file[histname].values()

        

"""
ClassLoader Class definition
"""

import uproot
import math
import numpy as np


class ClassLoader:

    def __init__(self, data_type, files, class_label):
        self.data_type = data_type
        self.files = files
        self.num_events = 0
        for batch in uproot.iterate(files, step_size=1000000, filter_name="TauJets.mcEventWeight"):
            self.num_events += len(batch["TauJets.mcEventWeight"])
        self.specific_batch_size = 0
        self.batches = None
        if __name__ == '__main__':
            self.class_label = class_label

    def load_batches(self, variable_list, total_num_events, total_batch_size):
        self.specific_batch_size = math.ceil(total_batch_size * self.num_events / total_num_events)
        self.batches = [batch for batch in uproot.iterate(self.files, filter_name=variable_list, step_size=100000,
                                                          library="ak", how="zip")]

    def batch_length(self, idx):
        return len(self.batches[idx]["TauJets"]["mcEventWeight"])

    def batch_labels(self, idx):
        return np.ones((len(self.batches[idx]["TauJets"]["mcEventWeight"]))) * self.class_label

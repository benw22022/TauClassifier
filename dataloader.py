"""
Dataloader class
"""

import uproot
import random

class DataLoader:

    def __init__(self, filename, features, batch_size=10000):
        self._filename = filename
        self._features = features
        self._batch_size = batch_size
        self._position = 0
        self._shuffled = False

        if isinstance(features, dict):
            tmp_arr = []
            for key in features:
                for i in range(0, len(features[key])):
                    tmp_arr.append(features[key][i])
            self._lazy_array = uproot.lazy(filename, step_size=batch_size, filter_name=tmp_arr)
        else:
            self._lazy_array = uproot.lazy(filename, step_size=batch_size, filter_name=features)
        self._length = len(self._lazy_array)

    def features(self):
        return self._features

    def batch_size(self):
        return self._batch_size

    def length(self):
        return self._length

    def apply_transform(self, variable, transform):
        return transform(self._lazy_array[variable])

    def get_batch(self, start_pos=0):
        return self._lazy_array[start_pos, start_pos + self._batch_size]

    def get_batch_cache(self):
        return self._lazy_array.cache

    def _next_position(self):
        if self._position < self._length - self._batch_size:
            self._position += self._batch_size

    def _reset_position(self):
        self._position = 0

    def get_next_batch(self):
        batch = self.get_batch(start_pos=self._position)
        self._next_position()
        return batch

    def lazy_array(self):
        return self._lazy_array

    def array(self, feature):
        return self._lazy_array[feature]

    def arrays(self, features):
        return self._lazy_array[features]

    def shuffle(self):
        rdm_index_list = list(range(0, len(self._lazy_array)))
        random.shuffle(rdm_index_list)
        self._lazy_array["index"] = rdm_index_list
        self._shuffled = True


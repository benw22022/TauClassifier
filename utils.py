"""
Utilities
___________________________________________
File containing useful functions and classes
"""

import time
from enum import Enum
from functools import total_ordering
from datetime import datetime
import os
from inspect import getframeinfo, stack
import tracemalloc
import glob
import numpy as np

@total_ordering
class LogLevels(Enum):
    ERROR = 0
    WARNING = 1
    INFO = 2
    DEBUG = 3
    HELPME = 4

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class Logger:

    def __init__(self, log_level='INFO'):
        self._start_time = time.time()
        self._log_level = LogLevels[log_level]
        self._log_file = open("data\\train.log", 'w')
        tracemalloc.start()

    def log(self, message, level='INFO'):
        if LogLevels[level] <= self._log_level:
            time_now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            caller = getframeinfo(stack()[1][0])
            filename = caller.filename
            line_num = caller.lineno
            filename = os.path.basename(filename)
            log_message = f"{time_now} {filename}:{line_num} {level} - {message}"
            print(log_message)
            self._log_file.write(f"{log_message}\n")

    def set_log_level(self, level):
        self._log_level = LogLevels[level]

    def log_memory_usage(self, level='DEBUG'):
        current, peak = tracemalloc.get_traced_memory()
        message = f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB"
        self.log(message, level)

    def __del__(self):
        self._log_file.close()


class FileHandler:
    """
    A class to handle files
    Can be sliced and printed
    """
    def __init__(self, label, search_dir, class_label=0):
        """
        Constructor
        :param label: A string to label the file handler
        :param search_dir: A string that can be passed to glob.glob that will grab the desired files
        :param class_label: A label that can be used to class the data - default is 0
        """
        self.label = label
        self.file_list = glob.glob(search_dir)
        self.class_label = class_label

    def __getitem__(self, key):
        """
        Overloads the [] operator
        :param key: An integer or slice
        :return: A new file handler object containing only the files at the requested index/slice
        """
        new_file_handler = FileHandler(self.label, "", self.class_label)
        if isinstance(key, slice):
            indices = range(*key.indices(len(self.file_list)))
            new_file_handler.file_list = [self.file_list[i] for i in indices]
            return new_file_handler
        new_file_handler.file_list = [self.file_list[key]]
        return new_file_handler

    def __str__(self):
        """
        Overloads the str() function - converts the file handler object to a string
        This is used to effectively override the print() function
        :return: A string detailing the properties of the file handler for when printing the object
        """
        if len(self.file_list) == 0:
            return f"FileHandler object: {self.label} has no associated files"

        ret_str = f"FileHandler object: {self.label} has {len(self.file_list)}"
        ret_str += f" files class label = {self.class_label}"
        ret_str += "\nFound these files:"
        for file in self.file_list:
            ret_str += f"\n{file}"
        return ret_str

def find_anomalous_entries(array, thresh, logger, arr_name=""):
    """
    Debugging function to look for strange entries - useful when trying to understand why loss is looking weird
    :param array: Array to look through
    :param thresh: Look for entries larger than this value
    :param logger: A Logger
    :return:
    """

    new_array = array[array > thresh]
    if len(new_array) > 0:
        logger.log(f"{arr_name} -  Found {len(new_array)} entries larger than thresh")
        logger.log(f"{arr_name} -  {new_array}")

    nan_arr = np.where(np.isnan(array))
    if len(nan_arr) > 0:
        logger.log(f"{arr_name} - found {len(nan_arr)} NaN values {nan_arr}")

    inf_arr = np.where(np.isinf(array))
    if len(inf_arr) > 0:
        logger.log(f"{arr_name} - found {len(inf_arr)} NaN values")



# Initialize logger as global variable
logger = Logger()


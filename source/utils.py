"""
Utilities
___________________________________________________________________
File containing useful functions and classes
Key objects:
Logger: a logging class
logger: a global instance of Logger shared between all code
FileHandler: A class to make the handling of file list easier
"""

import time
import uproot
from tqdm import tqdm
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
from functools import total_ordering
from datetime import datetime
import os
from inspect import getframeinfo, stack
import glob
import numpy as np
import tracemalloc
import getpass
import sys
tracemalloc.start()


@total_ordering
class LogLevels(Enum):
    """
    An enum class to help set logging levels
    Used in conjunction with Logger
    """
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

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def __init__(self, log_level='INFO'):
        """
        Constructor for a basic logging tool
        :param log_level (string) - sets the logging level
        """
        self._start_time = time.time()
        self._log_level = LogLevels[log_level.upper()]

    def log(self, message, level='INFO', log_mem=False):
        """
        Logging function. Writes message to terminal in the format
        <date> <time> <file>:<line> <log level> - <message> 
        :param message (str): message to be written to terminal
        :param level (str): string corresponding to enum
        :param log_mem (bool - default=False): If True will print current memory usage
        """
        if LogLevels[level] <= self._log_level:
            time_now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            caller = getframeinfo(stack()[1][0])
            filename = caller.filename
            line_num = caller.lineno
            filename = os.path.basename(filename)
            log_message = f"{time_now} {filename}:{line_num} {self.colour_level(level)} - {message}"
            if log_mem:
                current, peak = tracemalloc.get_traced_memory()
                message = f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB"
                log_message += f" - {message}"

            print(log_message)

    def set_log_level(self, level):
        self._log_level = LogLevels[level]

    def log_memory_usage(self, level='DEBUG'):
        current, peak = tracemalloc.get_traced_memory()
        message = f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB"
        self.log(message, level)

    def timer_start(self):
        self._start_time = time.time()
    
    def log_time(self, message, level='INFO'):
        delta_time = str(timedelta(seconds=time.time()-self._start_time))
        self.log(f"{message} in time {delta_time}", level)
        return delta_time

    def colour_level(self, level):
        if level == "INFO":
            return f"{self.OKBLUE}{level}{self.ENDC}"
        elif level == "DEBUG":
            return f"{self.OKCYAN}{level}{self.ENDC}"
        elif level == "HELPME":
            return f"{self.OKGREEN}{level}{self.ENDC}"
        elif level == "WARNING":
            return f"{self.WARNING}{level}{self.ENDC}"
        elif level == "ERROR":
            return f"{self.FAIL}{level}{self.ENDC}"
        return f"{self.OKBLUE}{level}{self.ENDC}"

# Initialize logger as global variable
logger = Logger()

class FileHandler:
    """
    A class to handle files
    Can be sliced and printed
    Stores additional useful data
    """
    def __init__(self, label, search_dir, class_label=0, cuts=None):
        """
        Constructor
        :param label: A string to label the file handler
        :param search_dir: A string that can be passed to glob.glob that will grab the desired files
        :param class_label: A label that can be used to class the data - default is 0
        :param cuts: A string of cuts that can be parsed by uproot.iterate
        """
        self.label = label
        self.file_list = glob.glob(search_dir)
        self.class_label = class_label
        self.cuts = cuts

    def __getitem__(self, key):
        """
        Overloads the [] operator
        Action slices the file list and returns a new FileHandler with the sliced list
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
        logger.log(f"{arr_name} - found {len(inf_arr)} infinite values {inf_arr}")


class Result:
    """
    Class to hold a batch of data - tbh never really used it; leaving it in because it has some
    nice slicing __getitem__
    """
    def __init__(self, track_arr, nPFO_arr, sPFO_arr, ctrack_arr, jets_arr, labels_arr, weights_arr):
        self.tracks = track_arr
        self.neutral_PFOs = nPFO_arr
        self.shot_PFOs = sPFO_arr
        self.conv_tracks = ctrack_arr
        self.jets = jets_arr
        self.labels = labels_arr
        self.weights = weights_arr

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, key):
        new_result = Result(self.tracks, self.neutral_PFOs, self.shot_PFOs, self.conv_tracks, self.jets, self.labels, self.weights)
        if isinstance(key, slice):
            indices = range(*key.indices(len(self.weights)))
            new_result.tracks = [self.tracks[i] for i in indices]
            new_result.neutral_PFOs = [self.neutral_PFOs[i] for i in indices]
            new_result.shot_PFOs = [self.shot_PFOs[i] for i in indices]
            new_result.conv_tracks = [self.conv_tracks[i] for i in indices]
            new_result.jets = [self.jets[i] for i in indices]
            new_result.labels = [self.labels[i] for i in indices]
            new_result.weights = [self.weights[i] for i in indices]
            return new_result
        new_result.tracks = self.tracks[key]
        new_result.neutral_PFOs = self.neutral_PFOs[key]
        new_result.shot_PFOs = self.shot_PFOs[key]
        new_result.conv_tracks = self.conv_tracks[key]
        new_result.jets = self.jets[key]
        new_result.labels = self.labels[key]
        new_result.weights = self.weights[key]
        return new_result

def get_best_weights(search_dir="network_weights"):
    """
    Search a directory of weights files and find the one from the last epoch saved
    Weights files must have naming convention along the lines of weights-<epoch>.h5
    :param search_dir (optional, default="network_weights"): Directory to search weight files for 
    """
    avail_weights = glob.glob(os.path.join(search_dir, "*.h5"))
    best_weights_num = 0 
    best_weights = ""
    for fname in avail_weights:
        weights_num = float(fname.split("-")[1].split(".")[0])
        if weights_num > best_weights_num:
            best_weights_num = weights_num
            best_weights = fname
    return best_weights


def get_number_of_events(fh_list):
    """
    Given a list of FileHandler Objects computes the number of events belonging to each class
    """
    njets = n1p0n = n1p1n = n1pXn = n3p0n = n3p1n = 0

    for fh in tqdm(fh_list):
        data = uproot.concatenate(fh.file_list, filter_name="TauJets.truthDecayMode", library='np')
        data = data["TauJets.truthDecayMode"]
        if fh.label == "Gammatautau":
            n1p0n += np.count_nonzero(data == 0)
            n1p1n += np.count_nonzero(data == 1)
            n1pXn += np.count_nonzero(data == 2)
            n3p0n += np.count_nonzero(data == 3)
            n3p1n += np.count_nonzero(data == 4)
        else:
            njets += len(data)

    return njets, n1p0n, n1p1n, n1pXn, n3p0n, n3p1n


def none_or_int(value):
    """
    A little function that will return None if parsed 'None' or a int. Used to parse -prong argument 
    :param value: A string tha
    """
    if value == 'None':
        return None
    return int(value)

def bytes_to_human(n_bytes):
    """
    Convert bytes to a human readable string
    """
    if n_bytes < 1e3:
        return f"{n_bytes:.2f} B"
    elif 1e3 <= n_bytes < 1e6:
        return f"{n_bytes / 1e3:.2f} kB"
    elif 1e6 <= n_bytes < 1e9:
        return f"{n_bytes / 1e6:.2f} MB"
    elif 1e9 <= n_bytes < 1e12:
        return f"{n_bytes / 1e9:.2f} GB"
    elif 1e12 <= n_bytes:
        return f"{n_bytes / 1e9:.2f} TB"


def profile_memory(obj, level='DEBUG'):
    """
    Get memory used by each class member
    :param obj: A class instance to get memory profile of 
    :param level (optional, default=DEBUG): Set logging level
    """
    members = [attr for attr in dir(obj) if not callable(getattr(obj, attr)) and not attr.startswith("__")]
    memory_dict = dict.fromkeys(members)

    for class_member in members:
        member_memory = sys.getsizeof(getattr(obj, class_member))
        memory_dict[class_member] = member_memory
    total_memory = sum(memory_dict.values())

    human_readable_mem_dict = memory_dict
    for key in human_readable_mem_dict:
        human_readable_mem_dict[key] = bytes_to_human(human_readable_mem_dict[key]) 

    logger.log(f"Total memory consumed by {obj}: {total_memory / 1e9} GB", level=level)

    return human_readable_mem_dict


def make_run_card(args):

    now = datetime.now()

    dirname = os.path.dirname(__file__)
    train_script = os.path.join(dirname, "run", "train.py")

    with open(os.path.join(f"{args.weights_save_dir}", "MetaData.dat"), 'w') as file:
        file.write("Training config")
        file.write(now.strftime("%d/%m/%Y %H:%M:%S"))
        file.write("\n")
        file.write(args.run_mode)
        file.write(f"Prong arguement is {args.prong}")
        file.write(f"Learning rate: {args.lr}")
        file.write(f"Model weights saved to {args.weights_save_dir}")
        file.write("Training script run is: \n")
        with open(train_script, 'r') as script: 
            for line in script: 
                file.write(line)

        file.write("\n\n")
        file.write("Model config is: ")

"""
Utilities
___________________________________________
File containing useful functions and classes
"""

import time
import tracemalloc
from enum import Enum
from functools import total_ordering
from datetime import datetime
import sys
import os

@total_ordering
class LogLevels(Enum):
    ERROR = 0
    WARNING = 1
    INFO = 2
    DEBUG = 3

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class Logger:

    def __init__(self, log_level='INFO'):
        self.start_time = time.time()
        self.log_level = LogLevels[log_level]

    def log(self, message, level='INFO'):
        if LogLevels[level] <= self.log_level:
            time_now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f"{time_now} {os.path.relpath(sys.argv[0])}: {level} - {message}")

# Initialize logger as gloabl variable
logger = Logger(log_level='INFO')


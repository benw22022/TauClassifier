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
logger = Logger(log_level='DEBUG')

class TimeMemoryMonitor:

    def __init__(self):
        self._start_time = time.time()
        tracemalloc.start()
        self._time_log = {"Start Time": 0}
        self._mem_log = {"Start Mem": tracemalloc.get_traced_memory()}

    def get_current_usage(self):
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
        print("--- %s seconds ---" % (time.time() - self._start_time))

    def log_time(self, log_message):
        self._time_log = {**self._time_log, **{log_message: time.time() - self._start_time}}

    def log_memory(self, log_message):
        self._mem_log = {**self._mem_log, **{log_message: tracemalloc.get_traced_memory()}}

    def log(self, log_message):
        self.log_time()
        self.log_memory()

    def get_time_log(self):
        return self._time_log

    def get_memory_log(self):
        return self._mem_log

    def get_logs(self):
        return self._time_log, self._mem_log

    def __del__(self):
        tracemalloc.stop()


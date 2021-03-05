"""
Utilities
___________________________________________
File containing useful functions and classes
"""

import time
import tracemalloc


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


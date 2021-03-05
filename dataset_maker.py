"""
dataset maker
______________
Functions to make dataset from MxAODs
Code works by using uproot.iterate to load batches of data, extracting arrays containing variables of interest
"""

import numpy as np
import random
import os


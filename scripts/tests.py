"""
A file to store performance tests
"""
import time
from utils import logger


def batch_loading_test(training_batch_generator):
    start_time = time.time()
    for i in range(0, len(training_batch_generator)):
        shape_trk, shape_conv_trk, shape_shot_pfo, shape_neut_pfo, shape_jet, _, _ = training_batch_generator.get_batch_shapes()

        print(f"Track shape = {shape_trk}   Conv Track shape = {shape_conv_trk}  Shot PFO shape = {shape_shot_pfo} "
              f"Neutral PFO shape = {shape_neut_pfo}  Jet shape = {shape_jet}")

    logger.log(f"Loaded all batches in {time.time() - start_time}")
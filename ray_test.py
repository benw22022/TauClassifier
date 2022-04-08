"""
Loop over generator
________________________________________________
Script to just loop over batch generator for
debugging 
"""
import os
import glob
import tqdm
# from source.datagenerator import DataGenerator
from source.ray_datagenerator import DataGenerator
from pympler import tracker
memory_tracker = tracker.SummaryTracker()
import awkward as ak
import uproot
import glob
import ray
import os
import psutil

def print_mem():
    current_process = psutil.Process(os.getpid())
    mem = current_process.memory_percent()
    for child in current_process.children(recursive=True):
        mem += child.memory_percent()
    print(f"Memory usage = {mem:2.2f} %")
    return mem

def grab_files(directory, glob_exprs):
    files = []
    for expr in glob_exprs: 
        files.append([[glob.glob(os.path.join(directory, expr, "*.root"))][0]])
        
    return files

def loop_test():

    files = glob.glob("../split_NTuples/*/*.root")

    from pympler import classtracker
    tr = classtracker.ClassTracker()
    tr.track_class(DataGenerator)
    tr.create_snapshot()

    training_batch_generator = DataGenerator(files, "config/features.yaml", batch_size=256, ncores=1, nsplits=30)

    
    while True:
        for i in tqdm.tqdm(range(0, len(training_batch_generator))):
            x = training_batch_generator[i]
            print(x)
        training_batch_generator.on_epoch_end()
        # mem = print_mem()
        # if mem > 20:
        #     training_batch_generator.reload()
        
        # training_batch_generator.print_diff()
        # memory_tracker.print_diff()
        # tr.create_snapshot()
        # tr.stats.print_summary()


if __name__ == "__main__":
    ray.init()
    loop_test()
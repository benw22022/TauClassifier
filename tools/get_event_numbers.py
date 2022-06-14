"""
Work out how many events are in each file
"""

import os
import glob
import uproot

NTUPLE_DIR = "../split_NTuples"

def get_breakdown(files):

    result = {}
    nevents_tot = 0
    for file in files:
        array = uproot.lazy(file, filter_name="TauJets.mcEventWeight")
        result[file] = len(array)
        nevents_tot += len(array)
    
    for file, nevents in result.items():
        print(f"{file}: {nevents} Events  -- {nevents / nevents_tot * 100: 1.2f} %")


if __name__ == "__main__":

    print("\n\n")
    get_breakdown(glob.glob(os.path.join(NTUPLE_DIR, "*Gammatautau*", "*.root")))
    print("\n\n")
    get_breakdown(glob.glob(os.path.join(NTUPLE_DIR, "*JZ1*", "*.root")))
    print("\n\n")
    get_breakdown(glob.glob(os.path.join(NTUPLE_DIR, "*JZ2*", "*.root")))
    print("\n\n")
    get_breakdown(glob.glob(os.path.join(NTUPLE_DIR, "*JZ3*", "*.root")))
    print("\n\n")
    get_breakdown(glob.glob(os.path.join(NTUPLE_DIR, "*JZ4*", "*.root")))
    print("\n\n")
    get_breakdown(glob.glob(os.path.join(NTUPLE_DIR, "*JZ5*", "*.root")))
    print("\n\n")
    get_breakdown(glob.glob(os.path.join(NTUPLE_DIR, "*JZ6*", "*.root")))
    print("\n\n")
    get_breakdown(glob.glob(os.path.join(NTUPLE_DIR, "*JZ7*", "*.root")))
    print("\n\n")
    get_breakdown(glob.glob(os.path.join(NTUPLE_DIR, "*JZ8*", "*.root")))
"""
Work out how many events are in each file
"""

import os
import glob
import uproot

NTUPLE_DIR = "../split_NTuples"

def get_breakdown(files, cut=None, just_total=True):

    
    result = {}
    nevents_tot = 0
    for file in files:
        array = uproot.concatenate(file, filter_name=["TauJets.mcEventWeight", "TauJets_truthDecayMode"], cut=cut)
        result[file] = len(array)
        nevents_tot += len(array)
    
    if not just_total:
        for file, nevents in result.items():
            print(f"{file}: {nevents} Events  -- {nevents / nevents_tot * 100: 1.2f} %")
    else:
        print(f"Total Number of events {nevents_tot}")
    

if __name__ == "__main__":

    print("\n\n")
    get_breakdown(glob.glob(os.path.join(NTUPLE_DIR, "*Gammatautau*", "*.root")))
    print("\n")
    print("# 1p0n")
    get_breakdown(glob.glob(os.path.join(NTUPLE_DIR, "*Gammatautau*", "*.root")), cut="TauJets_truthDecayMode == 0")
    
    print("\n")
    print("n1p1n")
    get_breakdown(glob.glob(os.path.join(NTUPLE_DIR, "*Gammatautau*", "*.root")), cut="TauJets_truthDecayMode == 1")
    
    print("\n")
    print("# 1pXn")
    get_breakdown(glob.glob(os.path.join(NTUPLE_DIR, "*Gammatautau*", "*.root")), cut="TauJets_truthDecayMode == 2")
    
    print("\n")
    print("# 3p0n")
    get_breakdown(glob.glob(os.path.join(NTUPLE_DIR, "*Gammatautau*", "*.root")), cut="TauJets_truthDecayMode == 3")
    
    print("\n")
    print("# 3pXn")
    get_breakdown(glob.glob(os.path.join(NTUPLE_DIR, "*Gammatautau*", "*.root")), cut="TauJets_truthDecayMode == 4")
    
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
"""
Split Root Files
________________________________________________________________
Split root files into uniform length files for more reliable 
train/test/val split.
Also add the branch to combine tau ID and decay mode labels
"""

import ROOT
import os
import tqdm
import math
from typing import List
import pandas as pd

ROOT.gInterpreter.Declare('''
ROOT::RVec<int> GenerateLabels(const int& decay_mode, const bool& is_jet)
{
    ROOT::RVec<int> label = {0, 0, 0, 0, 0, 0};
    if(is_jet == true){
        label[0] = 1;
    }
    else{
        label[decay_mode + 1] = 1;
    }
    return label;
}
''')

ROOT.gInterpreter.Declare('''
template<typename T>
ROOT::RVec<T> normalize(const ROOT::RVec<T>& x, const float& mean, const float& std_dev)
{
    ROOT::RVec<T> x_normed(x);
    
    for(int i{0}; i < x.size(); i++)

        x_normed = (x - mean) / (std_dev + 1e-8);

    return x_normed;
}
''')

ROOT.gInterpreter.Declare('''
template<typename T>
T normalize(const T& x, const float& mean, const float& std_dev)
{
    T x_normed(x);
    x_normed = (x - mean) / (std_dev + 1e-8);

    return x_normed;
}
''')

def split_files(infiles: List[str], outdir: List[str], outfile_tmplt: str, events_per_file: int=100000, is_fake: bool=False, treename: str="tree"):

    df = ROOT.RDataFrame("tree", infiles)
    df = df.Define("TauClassifier_Labels", f"GenerateLabels(TauJets.truthDecayMode, {str(is_fake).lower()})")
    nevents = df.Count().GetValue()

    stats_df = pd.read_csv("config/stats_df.csv", index_col=0)

    columns = [str(c) for c in df.GetColumnNames()]
    for column in columns:
        try:
            column_fixed = column.replace(".", "_")
            mean = stats_df.loc[column_fixed]["Mean"]
            std = stats_df.loc[column_fixed]["StdDev"]
            df = df.Define(f"{column_fixed}_normed", f"normalize({column}, {mean}, {std})")
            print(column_fixed)
        except KeyError:
            print(f"No key called {column_fixed}")


    try:
        os.makedirs(outdir)
    except OSError:
        pass

    pos = 0
    pbar = tqdm.tqdm(range(0, math.ceil(nevents / events_per_file)))
    for i in pbar:
        outfile_name = f"{outfile_tmplt}_{i:02d}.root"
        pbar.set_description(f"Processing {outfile_name}")

        outfile = os.path.join(outdir, outfile_name)

        df.Range(pos, pos + events_per_file).Snapshot(treename, outfile)
        pos += events_per_file

def main():
    
    tau_files = "../NTuples/user.bewilson.TauClassifierV3.425200.Pythia8EvtGen_A14NNPDF23LO_Gammatautau_MassWeight_v0_output.root/*.root"
    JZ1_files = "../NTuples/user.bewilson.TauClassifierV3.425200.Pythia8EvtGen_A14NNPDF23LO_Gammatautau_MassWeight_v0_output.root/*.root"
    JZ2_files = "../NTuples/user.bewilson.tauclassifier.364702.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_v0_output.root/*.root"
    JZ3_files = "../NTuples/user.bewilson.tauclassifier.364703.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3WithSW_v0_output.root/*.root"
    JZ4_files = "../NTuples/user.bewilson.tauclassifier.364704.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4WithSW_v0_output.root/*.root"
    JZ5_files = "../NTuples/user.bewilson.tauclassifier.364705.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ5WithSW_v0_output.root/*.root"
    JZ6_files = "../NTuples/user.bewilson.tauclassifier.364706.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ6WithSW_v0_output.root/*.root"
    JZ7_files = "../NTuples/user.bewilson.tauclassifier.364707.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ7WithSW_v0_output.root/*.root"
    JZ8_files = "../NTuples/user.bewilson.tauclassifier.364708.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ8WithSW_v0_output.root/*.root"

    # split_files(tau_files, "../split_NTuples/test", "Gammatautau", is_fake=False)
    # split_files(tau_files, "../split_NTuples/Gammatautau", "Gammatautau", is_fake=False)
    # Gammatautau done
    # split_files(JZ1_files, "../split_NTuples/JZ1", "JZ1", is_fake=True)
    # split_files(JZ2_files, "../split_NTuples/JZ2", "JZ2", is_fake=True)
    # split_files(JZ3_files, "../split_NTuples/JZ3", "JZ3", is_fake=True)
    # split_files(JZ4_files, "../split_NTuples/JZ4", "JZ4", is_fake=True, events_per_file=50000)
    split_files(JZ5_files, "../split_NTuples/JZ5", "JZ5", is_fake=True)
    # split_files(JZ6_files, "../split_NTuples/JZ6", "JZ6", is_fake=True)
    # split_files(JZ7_files, "../split_NTuples/JZ7", "JZ7", is_fake=True)
    # split_files(JZ8_files, "../split_NTuples/JZ8", "JZ8", is_fake=True)


if __name__ == "__main__":

   main()
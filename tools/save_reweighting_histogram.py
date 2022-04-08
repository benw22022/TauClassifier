"""
Save Re-Weighting Histogram
_________________________________________________________
Save a histogram for pT reweighting as a root file
"""

import ROOT
import glob
from typing import List, Tuple

ROOT.ROOT.EnableImplicitMT()

def get_reweight_hist(tau_files: List[str], fake_files: List[str], param: str="TauJets.ptJetSeed", nbins: int=500, range: Tuple[float, float]=(15e3, 1e7)) -> ROOT.TH1D:
    """
    Make a ratio histogram of tau pt / fake pt to reweight the ptJetSeed distribution of the
    fakes to match that of the taus. Default params are set up to reweight the ptJetSeed distribution
    but I've added options in case this changes
    args:
        tau_files: List[str] - A list of the tau files to use
        fake_files: List[str] - A list of the tau fakes (jets) files to use 
        param: str (default="TauJets.ptJetSeed") - parameter to reweight so that the histograms of taus & fakes match
        nbins: int (default=500) - Number of bins to use for the histograms
        range: Tuple[float, float] (default=(15e3, 1e7)) - Histogram range
    returns:
        ratio_hist: cppyy.gbl.TH1D - a ROOT TH1D representing the ratio of the tau hist / fake hist
    """

    tau_df = ROOT.RDataFrame("tree", tau_files)
    fake_df = ROOT.RDataFrame("tree", fake_files)

    tau_hist = tau_df.Histo1D(("Tau", "Tau", nbins, range[0], range[1]), param)
    fake_hist = fake_df.Histo1D(("Fake", "Fake", nbins, range[0], range[1]), param)

    ratio_hist = tau_hist.Clone("Tau / Jet")

    ratio_hist.Divide(fake_hist.GetPtr())

    return ratio_hist

def main():
    tau_files = glob.glob("../../NTuples/*Gammatautau*/*.root")
    fake_files = glob.glob("../../NTuples/*JZ*/*.root")

    reweight_hist = get_reweight_hist(tau_files, fake_files)

    outfile = ROOT.TFile.Open("../config/ReWeightHist.root", "RECREATE")

    outfile.WriteObject(reweight_hist, "ReWeightHist")

if __name__ == "__main__":

    main()

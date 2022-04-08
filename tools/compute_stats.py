"""
Compute Stats
___________________________________________________________________
Simple script to compute mean & std for every feature in dataset
"""

import ROOT
import pandas as pd
import glob 

ROOT.ROOT.EnableImplicitMT()

def compute_stats(file_list, verbose=True):
    """
    Computes mean and standard deviation of all columns from a list of root files
    args:
        file_list: List[str] -> a list of root files
        verbose: bool(True) -> Print results as computed
    returns:
        pd.DataFrame -> Pandas DataFrame with
                        rows: column names from root file
                        columns: ["Mean", "StdDeV"]
    """
    
    df = ROOT.RDataFrame("tree", file_list)

    columns = [str(c) for c in df.GetColumnNames()]

    stats = {}

    for column in columns:

        mean = float(df.Mean(column).GetValue())
        std = float(df.StdDev(column).GetValue())
        stats[column] = (mean, std) 

        if verbose:
            print(f"{column}: mean = {mean:0.3g}  std = {std:0.3g}")

    return pd.DataFrame.from_dict(stats, orient='index', columns=["Mean", "StdDev"])


if __name__ == "__main__":

    stats_df = compute_stats(glob.glob("../NTuples/*/*.root"))
    stats_df.to_csv("../config/stats_df.csv")
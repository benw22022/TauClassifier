import glob 
import tqdm
import uproot
import numpy as np
import pandas as pd
import awkward as ak

def get_mean_and_stddev(files, variable):

    data = uproot.concatenate(files, filter_name=variable, library='ak')[variable]

    try:
        data = ak.to_numpy(ak.flatten(data))
    except Exception:
        data = ak.to_numpy(data)

    mean = np.mean(data)
    std_dev = np.std(data)

    data = data[data - mean < 5 * std_dev]

    mean = np.mean(data)
    std_dev = np.std(data)

    return mean, std_dev


if __name__ == "__main__":

    files = glob.glob("/home/bewilson/NTuples/*/*.root")
    branches = uproot.open(f"{files[0]}:tree").keys()

    results = {}

    for variable in tqdm.tqdm(branches, total=len(branches)):
        
        if 'TauClusters' in variable:
            continue

        mean, std = get_mean_and_stddev(files, variable)
        
        results[variable] = (mean, std)

    df = pd.DataFrame.from_dict(results, orient='index', columns=["Mean", "StdDev"])
    df.to_csv("stats_df.csv")

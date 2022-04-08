from source import feature_handler as fh
import pandas as pd
import yaml

def make_feature_handler(features_yaml: str="config/features.yaml", stats_csv: str="config/stats_df.csv") -> fh.FeatureHandler():
    # Function to read yaml and csv files to initialize FeatureHandler

    stats_df = pd.read_csv(stats_csv, index_col=0)

    feature_handler = fh.FeatureHandler()

    with open(features_yaml, 'r') as stream:

        for branch, data in yaml.load(stream, Loader=yaml.FullLoader).items():

            feature_list = []
            for feature in data.features:
                
                mean = stats_df.iloc[list(stats_df.index).index(feature)]["Mean"]
                std = stats_df.iloc[list(stats_df.index).index(feature)]["StdDev"]
                feature_list.append(fh.Feature(feature, mean, std))

            feature_handler.add_branch(branch, feature_list, max_objects=data.max_objects)
            
    return feature_handler
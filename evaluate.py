import os
import tqdm 
import glob
import yaml
from sklearn.model_selection import train_test_split
from model.models import ModelDSNN
from source.dataloader import DataWriter


def evaluate():

    # Disable GPU (Don't really need it and it could cause issues if already training)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Load yaml config file 
    with open("config/file_config.yaml", 'r') as stream:
            file_config = yaml.load(stream, Loader=yaml.FullLoader)
    
    # Grab testing files
    tau_files = glob.glob(file_config["TauFiles"])
    jet_files = glob.glob(file_config["FakeFiles"])
    _, tau_test_files = train_test_split(tau_files, test_size=file_config["TestSplit"], random_state=file_config["RandomSeed"])
    _, jet_test_files = train_test_split(jet_files, test_size=file_config["TestSplit"], random_state=file_config["RandomSeed"])

    # TODO: Don't hard code this - add it as arg before
    # Load model
    weights_file = "network_weights/weights-02.h5"
    model = ModelDSNN("config/model_config.yaml", "config/features.yaml")
    model.load_weights(weights_file)

    # TODO: see if this can be parallelised with ray (May not work due to how fussy tf can be with model objects)
    # Write results to file
    for i, file in tqdm.tqdm(enumerate(tau_test_files)):
        loader = DataWriter(file, "config/features.yaml", batch_size=1000, step_size=10000)
        loader.write_results(model, output_file=f"results/taus_{i:02d}.root")
    
    for i, file in tqdm.tqdm(enumerate(jet_test_files)):
        loader = DataWriter(file, "config/features.yaml", batch_size=1000, step_size=10000)
        loader.write_results(model, output_file=f"results/jets_{i:02d}.root")


if __name__ == "__main__":
    evaluate()

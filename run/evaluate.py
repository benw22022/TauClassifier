import os
import tqdm 
import glob
import yaml
import ray
from sklearn.model_selection import train_test_split
from model.models import ModelDSNN
from source.datawriter import RayDataWriter, DataWriter

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
    # TODO: ^^^ Cannot pass model object to ray Actor - have to create model on Actor instantiation
    # Write results to file

    for i, file in tqdm.tqdm(enumerate(tau_test_files), total=len(tau_test_files)):
        loader = DataWriter(file, "config/features.yaml")
        loader.write_results(model, output_file=f"results/taus_{i:02d}.root")

    for i, file in tqdm.tqdm(enumerate(jet_test_files), total=len(jet_test_files)):
        loader = DataWriter(file, "config/features.yaml")
        loader.write_results(model, output_file=f"results/jets_{i:02d}.root")

    # results = []
    # for i, file in tqdm.tqdm(enumerate(tau_test_files)):
    #     loader = RayDataWriter.remote(file, "config/features.yaml")
    #     results.append(loader.write_results.remote(model, output_file=f"results/taus_{i:02d}.root"))
    # ray.get(results)

    # results = []
    # for i, file in tqdm.tqdm(enumerate(jet_test_files)):
    #     loader = RayDataWriter.remote(file, "config/features.yaml")
    #     results.append(loader.write_results.remote(model, output_file=f"results/jets_{i:02d}.root"))
    # ray.get(results)

if __name__ == "__main__":
    evaluate()

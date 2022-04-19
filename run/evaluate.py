import logger
log = logger.get_logger(__name__)
import os
import tqdm 
import glob
import  source
from model.models import ModelDSNN
from omegaconf import DictConfig
import glob

def get_last_weights():
    """
    Get last weights file saved
    """
    avail_weights = glob.glob("outputs/*/*/network_weights/*.h5")
    return  max(avail_weights, key=os.path.getctime)

def evaluate(config: DictConfig) -> None:

    log.info()

    # Disable GPU (Don't really need it and it could cause issues if already training)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Grab files
    _, tau_test_files, _ = source.get_files(config, "TauFiles") 
    _, jet_test_files, _ = source.get_files(config, "JetFiles") 
    
    # Grab weights file - automatically select last created weights file unless specified
    try:
        weights_file = config.weights
    except AttributeError:
        weights_file = get_last_weights()

    # Load model
    model = ModelDSNN(config)
    model.load_weights(weights_file)

    # TODO: see if this can be parallelised with ray (May not work due to how fussy tf can be with model objects)
    # TODO: ^^^ Cannot pass model object to ray Actor - have to create model on Actor instantiation
    # Write results to file
    for i, file in tqdm.tqdm(enumerate(tau_test_files), total=len(tau_test_files)):
        loader = source.DataWriter(file, config)
        loader.write_results(model, output_file=f"results/taus_{i:02d}.root")

    for i, file in tqdm.tqdm(enumerate(jet_test_files), total=len(jet_test_files)):
        loader = source.DataWriter(file, config)
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

import logger
log = logger.get_logger(__name__)
import os
import tqdm 
import glob
import  source
from model.models import ModelDSNN
from omegaconf import DictConfig
import glob
from hydra.utils import get_original_cwd, to_absolute_path
from pathlib import Path


def get_weights(config: DictConfig) -> str:
    """
    Grabs weights file specified in config. If no weights available then function finds the 
    most recent weights hdf5 file saved
    args:
        config: DictConfig - Hydra config object
    returns:
        weights_file: str - Path to weights file to be loaded
    """
    try:
        weights_file = config.weights
        log.info(f"Loading weights from specified file: {weights_file}")
    except AttributeError:
        avail_weights = glob.glob(os.path.join(get_original_cwd(), "outputs", "train_output", "*", "network_weights", "*.h5"))
        weights_file = max(avail_weights, key=os.path.getctime)
        log.info(f"Loading weights from last created file: {weights_file}")
    return weights_file


def evaluate(config: DictConfig) -> None:

    log.info("Running Evaluation")

    # Disable GPU (Don't really need it and it could cause issues if already training)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Grab files
    _, tau_test_files, _ = source.get_files(config, "TauFiles") 
    _, jet_test_files, _ = source.get_files(config, "FakeFiles") 
    
    # Grab weights file - automatically select last created weights file unless specified
    weights_file = get_weights(config)
    
    run_dir = Path(weights_file).parents[1]
    output_dir = os.path.join(run_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = ModelDSNN(config)
    model.load_weights(weights_file)

    # TODO: see if this can be parallelised with ray (May not work due to how fussy tf can be with model objects)
    # TODO: ^^^ Cannot pass model object to ray Actor - have to create model on Actor instantiation
    # Write results to file
    for i, file in tqdm.tqdm(enumerate(tau_test_files), total=len(tau_test_files)):
        loader = source.DataWriter(file, config) 
        loader.write_results(model, output_file=os.path.join(output_dir, f"taus_{i:02d}.root"))

    for i, file in tqdm.tqdm(enumerate(jet_test_files), total=len(jet_test_files)):
        loader = source.DataWriter(file, config)
        loader.write_results(model, output_file=os.path.join(output_dir, f"jets_{i:02d}.root"))

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

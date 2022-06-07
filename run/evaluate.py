from typing import Dict
import logger
log = logger.get_logger(__name__)
import os
import tqdm 
import glob
import  source
from model.models import ModelDSNN
import omegaconf
from omegaconf import DictConfig
import glob
from hydra.utils import get_original_cwd, to_absolute_path
from pathlib import Path
import yaml


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
        weights_file = os.path.join(get_original_cwd(), config.network_weights)
        log.info(f"Loading weights from specified file: {weights_file}")
    except AttributeError:
        avail_weights = glob.glob(os.path.join(get_original_cwd(), "outputs", "train_output", "*", "network_weights", "*.h5"))
        weights_file = max(avail_weights, key=os.path.getctime)
        log.info(f"Loading weights from last created file: {weights_file}")
    return weights_file


def set_config_keys(master_config: DictConfig, previous_config: Dict) -> None:
    """
    Recursively loop through config and set keys
    """
    for k, v in previous_config.items():
        if isinstance(v, dict):
            try:
                set_config_keys(master_config[k], v)
            except omegaconf.errors.ConfigKeyError:
                log.warn(f"Could not set key '{k}'. Key not in hydra config")
        else:
            try:
                master_config[k] = v
            except omegaconf.errors.ConfigKeyError:
                log.warn(f"Could not set key '{k}'. Key not in hydra config")


def load_config(config: DictConfig, run_dir: str) -> None:
    """
    Load old model config - in case anything changed
    """
    previous_config_path = os.path.join(run_dir, '.hydra', 'config.yaml')
    with open(previous_config_path, "r") as stream:
        previous_config = yaml.safe_load(stream)
        set_config_keys(config, previous_config)


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
    
    # Load config
    load_config(config, run_dir)

    # Load model
    model = ModelDSNN(config)
    model.load_weights(weights_file)

    # TODO: see if this can be parallelised with ray (May not work due to how fussy tf can be with model objects)
    # TODO: ^^^ Cannot pass model object to ray Actor - have to create model on Actor instantiation
    # Write results to file
    total_n_files = len(tau_test_files) + len(jet_test_files)
    with tqdm.tqdm(total=total_n_files) as pbar:
        for i, file in enumerate(tau_test_files):
            loader = source.DataWriter(file, config, cuts=config.tau_cuts) 
            loader.write_results(model, output_file=os.path.join(output_dir, f"taus_{i:02d}.root"))
            pbar.update(1)

        for i, file in enumerate(jet_test_files):
            loader = source.DataWriter(file, config, cuts=config.fake_cuts)
            loader.write_results(model, output_file=os.path.join(output_dir, f"jets_{i:02d}.root"))
            pbar.update(1)

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

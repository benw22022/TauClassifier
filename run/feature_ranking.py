"""
Permutation Feature Importance
"""

import logger
log = logger.get_logger(__name__)
import os
import ray
import run
import source
from pathlib import Path
from run.evaluate import get_weights
from model.models import ModelDSNN
from source.permutation_data_generator import PermutationDataGenerator
from omegaconf import DictConfig
import pandas as pd
import tensorflow as tf


def feature_rank(config: DictConfig):

    log.info("Running feature ranking")

    # Initialise Ray
    ray.init(runtime_env={"py_modules": [source, run, logger]})

    # Grab weights file - automatically select last created weights file unless specified
    weights_file = get_weights(config)
    
    run_dir = Path(weights_file).parents[1]
    output_dir = os.path.join(run_dir, "ranking")
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = ModelDSNN(config)
    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=[tf.keras.metrics.CategoricalAccuracy()], )

    # Grab train/val files
    _, tau_test_files, _ = source.get_files(config, "TauFiles") 
    _, jet_test_files, _ = source.get_files(config, "FakeFiles") 

    # Generator
    testing_generator = PermutationDataGenerator(tau_test_files, jet_test_files, config, batch_size=1000000, step_size=config.step_size, name='TrainGen')

    # Get baseline
    baseline_loss, baseline_acc = model.evaluate(testing_generator)
    log.info(f"Baseline: Loss = {baseline_loss:.3f}\t Accuracy = {baseline_acc:.3f}")

    results_dict = {'feature': [], 'loss_delta': [], 'accuracy_delta': []}
    
    # Ranking
    for i, branch_name in enumerate(config.branches):
        for j, feature in enumerate(config.branches[branch_name].features):
            testing_generator.perm_index = (i, j)
            testing_generator.reset()
            loss, acc = model.evaluate(testing_generator)
            results_dict['feature'].append(feature)
            results_dict['loss_delta'].append(loss - baseline_loss)
            results_dict['accuracy_delta'].append(baseline_acc - acc)
            log.info(f"{branch_name} - {feature}: Loss = {loss:.3f}\t Accuracy = {acc:.3f}")
    
    # Sort table
    results_df = pd.DataFrame.from_dict(results_dict)
    results_df.sort_values(by=['loss_delta'])

    # Write results to file
    outfile = os.path.join(output_dir, "feature_ranking.csv")
    results_df.to_csv(outfile, index=False)
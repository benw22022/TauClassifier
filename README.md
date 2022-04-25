# Unified Tau Classifier
## Description
Development of a combined tau ID and tau decay mode classifier for ATLAS Tau Group


## Enviroment setup
Install miniconda\
  `conda env create -f config/environment.yaml`

## Dataset creation
TODO: Add documentation for THOR code

## Data preparation
To get efficient test/train/val splits data is split up into 100,000 event chunks and to improve network training features are normalised to have mean zero and standard deviation one.\
This part is **still work in progess** but for now do:\
`python3 tools/compute_stats.py` to get a csv of means and std devs\
`python3 tools/uproot_ntuple_writer.py` to write out the files

## How to run
The [Hydra](https://hydra.cc/) package is used to parse command line arguements and manage config settings
To avoid having to shuffle the entire dataset together, sub-batches of data for taus and fakes are loaded seperately and merged to create a complete training batch. To speed this up, the dataloading is run on a pair of parallel python processes using the [Ray](https://www.ray.io/) library.
### Training
To run training do\
`python3 tauclassifier.py learning_rate=1e-4 batch_size=1024`\
Training related options can be found in `config/training_config.yaml`\
Each training run will be saved to its own working directory saved in `outputs/train_output/<date-time>`

### Evaluation
To save results of a training run\
`python3 tauclassifier.py run_mode=evaluate`\
This will automatically select the most recently saved weights file\
To select a specific weights file add a `weights` field to config e.g.\
`python3 tauclassifier.py evaluate +weights=outputs/train_output/2022-04-20_10-50-12/network_weights/weights-05.h5`

### Visualisation
To plot performance plots from a run \
`python3 tauclassifier.py run_mode=visualise`\
To run on a specific set of evaluated ntuples add `results` field to config e.g.\
`python3 tauclassifier.py run_mode=visualise +results=outputs/train_output/2022-04-20_10-50-12/network_weights/weights-05.h5`\
Options related to evaluation and visualisation steps are found in `config/evaluation_config.yaml`\

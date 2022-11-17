# Unified Tau Classifier
## Description
Development of a combined tau ID and tau decay mode classifier for ATLAS Tau Group
See dev branch for latest commits

## Enviroment setup
Install miniconda\
  `conda env create -f config/environment.yaml`

## Dataset creation
My fork of [`THOR`](https://gitlab.cern.ch/atlas-perf-tau/THOR) can be found [here](https://gitlab.cern.ch/atlas-perf-tau/THOR). 
You'll also need my fork of [`tauRecToolsDev`](https://gitlab.cern.ch/atlas-perf-tau/tauRecToolsDev) which you can get from [here](https://gitlab.cern.ch/atlas-perf-tau/tauRecToolsDev). When you setup `THOR` it will copy the central branch of `tauRecToolsDev`, you'll need to replace it with mine.
The AODs for the Tau Classifier dataset can be found in `THOR/THOR/share/datasets/MC20dDijet.txt` and `THOR/THOR/share/datasets/MC20dGammatautau.txt`, before submitting any jobs to the grid I highly recommend downloading a part of one of these datasets with `Rucio` to test that the code works locally.
To run over a local file do:
```
thor THOR/THOR/share/StreamNNDecayMode/Main.py -i <path-to-folder-with-data> -o "output"
```
Such a path might look like `"/afs/cern.ch/work/b/bewilson/public/data"`, where I have a gammatautau AOD sample that I downloaded sometime ago.
Once you've checked that your THOR joboption runs locally, to run the gamma* -> tau tau samples run:
```
thor THOR/THOR/share/StreamNNDecayMode/Main.py -r grid -g THOR/THOR/share/datasets/MC20dGammatautau.txt --gridstreamname TauClassifierV1 --gridrunversion 0 --nFilesPerJob=5
```
and likewise for the dijet samples
```
thor THOR/THOR/share/StreamNNDecayMode/Main.py -r grid -g THOR/THOR/share/datasets/MC20dDijet.txt --gridstreamname TauClassifierV1 --gridrunversion 0 --nFilesPerJob=5
```
You can monitor the progress of your grid jobs on the [bigpanda](https://bigpanda.cern.ch/user/) website. You may find that some proportion of your grid jobs fail, to fix this (assuming this isn't a fault in your code) by resubmitting the failed jobs or by experimenting with changing the `nFilesPerJob` option.

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
`python3 tauclassifier.py run=evaluate`\
This will automatically select the most recently saved weights file\
To select a specific weights file add a `weights` field to config e.g.\
`python3 tauclassifier.py evaluate +weights=outputs/train_output/2022-04-20_10-50-12/network_weights/weights-05.h5`

### Visualisation
To plot performance plots from a run \
`python3 tauclassifier.py run=visualise`\
To run on a specific set of evaluated ntuples add `results` field to config e.g.\
`python3 tauclassifier.py run=visualise +results=outputs/train_output/2022-04-20_10-50-12/network_weights/weights-05.h5`\
Options related to evaluation and visualisation steps are found in `config/evaluation_config.yaml`\

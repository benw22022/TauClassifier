# TauClassifier
Development of a combined tau ID and tau decay mode classifier for ATLAS Tau Group

TODO: Write a better README!
A work in progress - apologies for any confusing bits!
Documentation is unfortunatly still lacking as I work out the best way to implement features 
and fix bugs

Main commands:
python3 tauclassifier.py train    Train neural network
python3 tauclassifier.py test     Plot confusion matrix and ROC for test data

Additional run modes that are less polished / work in progress:
rank			Do permutation variable ranking
plot_variables            Plot input variables (needs improvement)
plot_previous	            Plots current tauID RNN ROC and decay mode classifier confusion matrix
scan                      Do a learning rate scan (still experimental)
evaluate                  Decorate NTuples with network scores (coming soon)

This code uses the Uproot library to load batches of data directly from Root NTuples rather than generating intermediary files such as HDF5
The Ray libray is used to read in multiple file streams in parallel to improve data loading times
Data loading is handled by several main classes:
1. FileHandler: A basic class to handle file lists
2. DataLoader: A class to handle an Uproot.iterator object. Loads and processes batches of data from a particular file stream
3. DataGenerator: A class to handle multiple DataLoaders running in parallel using Ray

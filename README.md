# TauClassifier
Development of a combined tau ID and tau decay mode classifier for ATLAS Tau Group

A work in progress - apologies for any confusing bits!

Code is run from the main steering script called tauclassifier.py
Modes supported:
1. train: Runs the training of the neural network
2. evaluate: Evalutes the predictions for all NTuples and stores the results to a npz file
3. test: Plots confusion matrix and ROC curve for the testing data
4. rank: Performs permutation variable ranking and saves the results to csv
5. scan: Performs a learning rate scan

This code uses the Uproot library to load batches of data directly from Root NTuples rather than generating intermediary files such as HDF5
The Ray libray is used to read in multiple file streams in parallel to improve data loading times
Data loading is handled by several main classes:
1. FileHandler: A basic class to handle file lists
2. DataLoader: A class to handle an Uproot.iterator object. Loads and processes batches of data from a particular file stream
3. DataGenerator: A class to handle multiple DataLoaders running in parallel using Ray

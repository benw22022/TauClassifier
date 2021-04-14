# TauClassifier
Development of a combined tau ID and tau decay mode classifier for ATLAS Tau Group

A work in progress - apologies for any confusing bits!
Uses a Keras data generator (DataGenerator.py) to feed neural networks directly from NTuples using uproot.lazy. Each data file type (e.g. Gammatautau, JZ1, JZ2, etc...) has a  
chunk of data loaded and a batch of training data is made by concatenating all the arrays together. This avoids having all the data loaded into memory at once.
The DataGenerator class works by:
  1. Making a Dataloader class for each file type. This is a helper class I made which stores the lazily loaded chunks and works out how many events should be in each chunk- 
     it also stores info on whether data is signal/background. This class can return a lazily loaded chunk of data from an index.
  2. When training for each batch the __getitem__ method is called with an index, which iterates up to a len set by the __len__ method
     - __get__item calls load_batch()
     - load_batch() makes an array for each variable type that gets fed to the network (tracks, conv-tracks, shot-pfo, neutral-pfo, jets, labels and weights). For each data type
     in its list of DataLoaders it calls _load_batch_from_data() to get a chunck of data which it then joins together and returns.
     - _load_batch_from_data(): For the variable types with nested objects (tracks, conv-tracks, shot-pfo, neutral-pfo) the pad_and_reshape_nested_arrays() method is called
     - pad_and_reshape_nested_arrays(): 
            - Concatenates all variable arrays of a specific variable type together to make one large ragged awkward array
            - Pads the array with Nones and clips it to a certain value so that the awkward array is now rectilinear
            - The awkward array is then convered to a numpy array
            - The numpy array is then reshaped to match the network input shape is -> (#Number of events, #Number of variables, #length of arrays (clipping value))
     - For variable types which are already recilinear and contain multiple variables (jets) the reshape_arrays() method is called which does a similar thing to the above except
       without the padding. The shape is also differnt -> (#Number of events, #Number of variables)
     - Added option to use pool.starmap to read chunks of data from each datafile in parallel - this significantly speeds up loading of batches of data 

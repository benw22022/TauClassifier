"""
Main Code Body
"""
from variables import variables_dictionary
from models import tauid_rnn_model
from DataGenerator import DataGenerator
from files import files_dictionary
import time
from utils import logger


def main():
      # Start timer

      logger.log("Beginning dataset preparation", 'INFO')

      # Initialize Generators

      cuts = {"Gammatautau": "TauJets.truthProng == 1"}

      training_batch_generator = DataGenerator(files_dictionary, variables_dictionary, batch_size=100000, cuts=cuts, )
      #testing_batch_generator = DataGenerator(X_test_idx, variables_dictionary)
      #validation_batch_generator = DataGenerator(X_val_idx, variables_dictionary)

      #training_batch_generator.write_lazy_arrays()
      start_time = time.time()

      for i in range(0, len(training_batch_generator)):
            shape_trk, shape_conv_trk, shape_shot_pfo, shape_neut_pfo, shape_jet, _, _ = training_batch_generator.get_batch_shapes()

            print(f"Track shape = {shape_trk}   Conv Track shape = {shape_conv_trk}  Shot PFO shape = {shape_shot_pfo} "
                  f"Neutral PFO shape = {shape_neut_pfo}  Jet shape = {shape_jet}")

      logger.log(f"Loaded all batches in {start_time - time.time()}")

      # Initialize Model
      #model = tauid_rnn_model(shape_trk[1:], shape_cls[1:], shape_jet[1:])
      #model.summary()
      #model.compile(optimizer="adam", loss="binary_crossentropy",  metrics=["accuracy"])

      # Train Model
      #history = model.fit(training_batch_generator, epochs=100, max_queue_size=4, use_multiprocessing=False, shuffle=True)

if __name__ == "__main__":

   main()
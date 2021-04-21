"""
Main Code Body
"""
import os
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'    # Acclerated Linear Algbra (XLA) Seems to actually make things slower
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                     # Disables GPU


from variables import variables_dictionary
from models import tauid_rnn_model, ModelDSNN
from DataGenerator import DataGenerator
from files import training_files_dictionary, validation_files_dictionary
import time
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from utils import logger


logger.set_log_level('INFO')


# custom callback for multi-gpu model saving
class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self, model, path, monitor='val_loss', verbose=1,
                 save_best_only=False, save_weights_only=True):
        self._model = model
        super(ParallelModelCheckpoint, self).__init__(path, monitor, verbose, save_best_only, save_weights_only)

    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self._model)


def main():
    logger.log("Beginning dataset preparation", 'INFO')

    # Initialize Generators
    cuts = {"Gammatautau": "TauJets.truthProng == 1"}
    training_batch_generator = DataGenerator(training_files_dictionary, variables_dictionary, nbatches=1500, cuts=cuts)

    # Initialize Model

    # Work out input shapes
    shape_trk, shape_conv_trk, shape_shot_pfo, shape_neut_pfo, shape_jet = None, None, None, None, None
    for i in range(0, len(training_batch_generator)):
        shape_trk, shape_neut_pfo, shape_shot_pfo, shape_conv_trk, shape_jet, shape_label, shape_weight = training_batch_generator.get_batch_shapes()
        logger.log(f"TauTracks Shape = {shape_trk}")
        logger.log(f"ConvTracks Shape = {shape_conv_trk}")
        logger.log(f"ShotPFO Shape = {shape_shot_pfo}")
        logger.log(f"NeutralPFO Shape = {shape_neut_pfo}")
        logger.log(f"TauJets Shape = {shape_jet}")
        logger.log(f"Labels Shape = {shape_label}")
        logger.log(f"Weight Shape = {shape_weight}")
        break
    training_batch_generator.reset_generator()

    config_dict = {"shapes":
                       {"TauTrack": shape_trk[1:],
                        "ConvTrack": shape_conv_trk[1:],
                        "ShotPFO": shape_shot_pfo[1:],
                        "NeutralPFO": shape_neut_pfo[1:],
                        "TauJets": shape_jet[1:],
                        },
                   "n_tdd":
                       {"TauTrack": 3,
                        "ConvTrack": 3,
                        "ShotPFO": 3,
                        "NeutralPFO": 4,
                        "TauJets": 3,
                        },
                   "n_h":
                       {"TauTrack": 3,
                        "ConvTrack": 3,
                        "ShotPFO": 3,
                        "NeutralPFO": 3,
                        "TauJets": 3,
                        },
                   "n_hiddens":
                       {"TauTrack": [20, 20, 20],
                        "ConvTrack": [20, 20, 20],
                        "ShotPFO": [20, 20, 20],
                        "NeutralPFO": [60, 40, 40],
                        "TauJets": [20, 20, 20],
                        },
                   "n_inputs":
                       {"TauTrack": [20, 20, 20],
                        "ConvTrack": [20, 20, 20],
                        "ShotPFO": [20, 20, 20],
                        "NeutralPFO": [80, 80, 60, 60],
                        "TauJets": [20, 20, 20],
                        },
                   "n_fc1": 100,
                   "n_fc2": 50,
                   "n_classes": 4,
                   }

    print(config_dict["shapes"])

    model = ModelDSNN(config_dict)
    validation_batch_generator = DataGenerator(validation_files_dictionary, variables_dictionary, nbatches=1500, cuts=cuts)

    # Configure callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss", min_delta=0.0001,
        patience=10, verbose=0, restore_best_weights=True)

    model_checkpoint = ParallelModelCheckpoint(model,
                                               path=os.path.join("data", 'weights-{epoch:02d}.h5'),
                                               monitor="val_loss", save_best_only=False, save_weights_only=True,
                                               verbose=0)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=3, min_lr=4e-6)

    callbacks = [early_stopping, model_checkpoint,  reduce_lr]

    model.summary()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Train Model
    history = model.fit(training_batch_generator, epochs=100, callbacks=callbacks,
                        validation_data=validation_batch_generator, validation_freq=1, verbose=1, shuffle=True,
                        )

    #max_queue_size=6, workers=6, use_multiprocessing=True

if __name__ == "__main__":
    main()

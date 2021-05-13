"""
Configurations and selections
_______________________________________________________________________________________________________________________
The file that contains all the configurations and selection criteria
TODO: This should really be done using YAML - user shouldn't need to edit .py files unnecessarily
"""

import tensorflow as tf

# Bowen's DSNN config dictionary
config_dict = {"shapes":
				   {"TauTrack": (10, 20),
					"NeutralPFO": (22, 20),
					"ShotPFO": (6, 20),
					"ConvTrack": (10, 20),
					"TauJets": (9),
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
			   "dropout": 0.025
			   }

# Cuts to apply to data
cuts = {"Gammatautau": "(TauJets.truthProng == 1) & (TauJets.ptJetSeed > 15000.0)",
		"JZ1": "TauJets.ptJetSeed > 15000.0",
		"JZ2": "TauJets.ptJetSeed > 15000.0",
		"JZ3": "TauJets.ptJetSeed > 15000.0",
		"JZ4": "TauJets.ptJetSeed > 15000.0",
		"JZ5": "TauJets.ptJetSeed > 15000.0",
		"JZ6": "TauJets.ptJetSeed > 15000.0",
		"JZ7": "TauJets.ptJetSeed > 15000.0",
		"JZ8": "TauJets.ptJetSeed > 15000.0",
		}

# Tensorflow output types
types = (
	(tf.float32,
	 tf.float32,
	 tf.float32,
	 tf.float32,
	 tf.float32),
	tf.float32,
	tf.float32)

# Tensorflow output shapes
shapes = (
	(tf.TensorShape([None, 10, 20]),
	 tf.TensorShape([None, 22, 20]),
	 tf.TensorShape([None, 6, 20]),
	 tf.TensorShape([None, 10, 20]),
	 tf.TensorShape([None, 9])),
	tf.TensorShape([None, 4]),
	tf.TensorShape([None])
)

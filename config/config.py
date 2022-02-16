"""
Configurations and selections
_______________________________________________________________________________________________________________________
The file that contains all the configurations and selection criteria
TODO: This should really be done using YAML - user shouldn't need to edit .py files unnecessarily
"""

import tensorflow as tf
from config.variables import variable_handler
import model.models

# Directory pointing to the NTuples to train/test on
ntuple_dir = "/home/bewilson/NTuples"

# Bowen's DSNN config dictionary
config_dict = {"shapes":
				   {"TauTrack": (len(variable_handler.get("TauTracks")),) + (3,),
					"NeutralPFO": (len(variable_handler.get("NeutralPFO")),) + (6,),
					"ShotPFO": (len(variable_handler.get("ShotPFO")),) + (8,),
					"ConvTrack": (len(variable_handler.get("ConvTrack")),) + (4,),
					"TauJets": (len(variable_handler.get("TauJets")),),
					},
			   "n_tdd":
				   {"TauTrack": 4,
					"ConvTrack": 4,
					"ShotPFO": 4,
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
				   {"TauTrack": [60, 40, 40],
					"ConvTrack": [60, 40, 40],
					"ShotPFO": [60, 40, 40],
					"NeutralPFO": [60, 40, 40],
					"TauJets": [20, 20, 20],
					},
			   "n_inputs":
				   {"TauTrack": [80, 80, 60, 60],
					"ConvTrack": [80, 80, 60, 60],
					"ShotPFO": [80, 80, 60, 60],
					"NeutralPFO": [80, 80, 60, 60],
					"TauJets": [20, 20, 20],
					},
			   "n_fc1": 100,
			   "n_fc2": 50,
			   "n_classes": 6,
			   "dropout": 0.025
			   }


"""""""""""""""""""""""""""""""""""""""
Cuts to apply to data
"""""""""""""""""""""""""""""""""""""""

def get_cuts(prong=None, decay_mode=None):
	common_cuts = "(TauJets.ptJetSeed > 15000.0) & (TauJets.ptJetSeed < 10000000.0)"
	prong_cut = ""
	dm_cut = ""
	
	if prong == 1:
		prong_cut = " & (TauJets.truthProng == 1)"
	if prong == 3:
		prong_cut = "& (TauJets.truthProng == 3)"
	
	if decay_mode is not None:
		dm_cut = f"& (TauJets.truthDecayMode == {decay_mode})"

	cuts_dict = {"Gammatautau": common_cuts + prong_cut + dm_cut,
				"JZ1": common_cuts,
				"JZ2": common_cuts,
				"JZ3": common_cuts,
				"JZ4": common_cuts,
				"JZ5": common_cuts,
				"JZ6": common_cuts,
				"JZ7": common_cuts,
				"JZ8": common_cuts,
				}
	return cuts_dict

models_dict = {"DSNN": model.models.ModelDSNN,
		  	   "SetTransformer": model.models.SetTransformer, 
			   "DSNN_2Step": model.models.ModelDSNN_2Step}

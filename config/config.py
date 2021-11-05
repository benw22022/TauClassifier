"""
Configurations and selections
_______________________________________________________________________________________________________________________
The file that contains all the configurations and selection criteria
TODO: This should really be done using YAML - user shouldn't need to edit .py files unnecessarily
"""

import tensorflow as tf
from config.variables import variables_dictionary
from model.models import ModelDSNN, SetTransformer

# Directory pointing to the NTuples to train/test on
ntuple_dir = "../NTuples"

# Bowen's DSNN config dictionary
config_dict = {"shapes":
				   {"TauTrack": (len(variables_dictionary["TauTracks"]),) + (10,),
					"NeutralPFO": (len(variables_dictionary["NeutralPFO"]),) + (10,),
					"ShotPFO": (len(variables_dictionary["ShotPFO"]),) + (10,),
					"ConvTrack": (len(variables_dictionary["ConvTrack"]),) + (10,),
					"TauJets": (len(variables_dictionary["TauJets"]),),
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
			   "n_classes": 6,
			   "dropout": 0.025
			   }


"""""""""""""""""""""""""""""""""""""""
Cuts to apply to data
"""""""""""""""""""""""""""""""""""""""

def get_cuts(prong=None):
	common_cuts = "(TauJets.ptJetSeed > 15000.0) & (TauJets.ptJetSeed < 10000000.0) & (TauJets.ptRatioEflowApprox < 5) & (TauJets.etOverPtLeadTrk < 30)"
	prong_cut = ""
	if prong == 1:
		prong_cut = " & (TauJets.truthProng == 1)"
	if prong == 3:
		prong_cut = "& (TauJets.truthProng == 3)"

	cuts_dict = {"Gammatautau": common_cuts + prong_cut,
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

models_dict = {"DSNN": ModelDSNN,
		  	   "SetTransformer": SetTransformer}

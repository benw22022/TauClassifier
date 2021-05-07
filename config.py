"""
File for configs
Note: Temp
"""

config_dict = {"shapes":
				   {"TauTrack": (14, 20),
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
			   }

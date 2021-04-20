"""
This is where the data pre-processing will go
"""

import numpy as np
from keras import backend as K

procs = {"Log": [

		"TauTracks.pt",
		"TauTracks.d0TJVA",
		"TauTracks.d0SigTJVA",
		"TauTracks.z0sinthetaTJVA",
		"TauTracks.z0sinthetaSigTJVA",
		"ConvTrack.pt",
		"ConvTrack.jetpt",
		"ConvTrack.d0TJVA",
		"ConvTrack.d0SigTJVA",
		"ConvTrack.z0sinthetaTJVA",
		"ConvTrack.z0sinthetaSigTJVA",
		"ShotPFO.pt",
		"ShotPFO.jetpt",
		"NeutralPFO.pt",
		"NeutralPFO.jetpt",
		"NeutralPFO.SECOND_R",
		"NeutralPFO.CENTER_LAMBDA",
		"NeutralPFO.SECOND_ENG_DENS",
		"NeutralPFO.ENG_FRAC_CORE",
		"NeutralPFO.NPosECells_EM1",
		"NeutralPFO.NPosECells_EM2",
		"NeutralPFO.energy_EM1",
		"NeutralPFO.energy_EM2",
		"NeutralPFO.EM1CoreFrac",
		"NeutralPFO.firstEtaWRTClusterPosition_EM1",
		"NeutralPFO.firstEtaWRTClusterPosition_EM2",
		"NeutralPFO.secondEtaWRTClusterPosition_EM1",
		"NeutralPFO.secondEtaWRTClusterPosition_EM2",
		"TauJets.etOverPtLeadTrk",
		"TauJets.dRmax",
		"TauJets.SumPtTrkFrac",
		"TauJets.ptRatioEflowApprox",
		"TauJets.ptIntermediateAxis",
		"TauJets.ptJetSeed",
		]
}

def finite_log(m):
	return  np.log2(m, out=np.zeros_like(m), where=(m > 1))
"""
List of input variables from MxAODs
TODO: Make this a yaml config file (Bad practice to use a .py file for configs)
"""
"""
input_variables = [
    "TauJetsAuxDyn.trk_pt",
    "TauJetsAuxDyn.trk_ptJetSeed",
    "TauJetsAuxDyn.trk_z0sinthetaTJVA",
    "TauJetsAuxDyn.trk_dEta",
    "TauJetsAuxDyn.trk_dPhi",
    "TauJetsAuxDyn.trk_nInnermostPixelHits",
    "TauJetsAuxDyn.trk_nPixelHits",
    "TauJetsAuxDyn.trk_nSCTHits",
    "TauJetsAuxDyn.cls_et",
    "TauJetsAuxDyn.cls_SECOND_R",
    "TauJetsAuxDyn.cls_SECOND_LAMBDA",
    "TauJetsAuxDyn.cls_CENTER_LAMBDA",
    "TauJetsAuxDyn.centFrac",
    "TauJetsAuxDyn.etOverPtLeadTrk",
    "TauJetsAuxDyn.dRmax",
    "TauJetsAuxDyn.SumPtTrkFrac",
    "TauJetsAuxDyn.ptRatioEflowApprox",
    "TauJetsAuxDyn.ptIntermediateAxis",
   # "EventInfoAuxDyn.mcEventWeights",
    "EventInfoAuxDyn.mcChannelNumber"
    ]

variables_dictionary = {"Tracks": ["TauJetsAuxDyn.trk_pt",
                        "TauJetsAuxDyn.trk_ptJetSeed",
                        "TauJetsAuxDyn.trk_z0sinthetaTJVA",
                        "TauJetsAuxDyn.trk_dEta",
                        "TauJetsAuxDyn.trk_dPhi",
                        "TauJetsAuxDyn.trk_nInnermostPixelHits",
                        "TauJetsAuxDyn.trk_nPixelHits",
                        "TauJetsAuxDyn.trk_nSCTHits"],
             "Clusters": ["TauJetsAuxDyn.cls_et",
                          "TauJetsAuxDyn.cls_SECOND_R",
                          "TauJetsAuxDyn.cls_SECOND_LAMBDA",
                          "TauJetsAuxDyn.cls_CENTER_LAMBDA"],
             "Jets": ["TauJetsAuxDyn.centFrac",
                      "TauJetsAuxDyn.etOverPtLeadTrk",
                      "TauJetsAuxDyn.dRmax",
                      "TauJetsAuxDyn.SumPtTrkFrac",
                      "TauJetsAuxDyn.ptRatioEflowApprox",
                      "TauJetsAuxDyn.ptIntermediateAxis"],
             "Weights": ["EventInfoAuxDyn.mcEventWeights"]
             }

"""
input_variables = [
# 8 Vars
"TauTracks.pt", # LOG
#"TauTracks.pt_jetseed", # LOG What- doesn't exist? Is "TauJets.ptJetSeed"
"TauJets.ptJetSeed", # LOG
"TauTracks.z0sinthetaTJVA", #ABS LOG (NOTE lowercase theta (change from R21)
"TauTracks.dEta",
"TauTracks.dPhi",
"TauTracks.nInnermostPixelHits",
"TauTracks.nPixelHits",
"TauTracks.nSCTHits",

# 7 vars
"TauClusters.et",
#"TauClusters.pt_jetseed", #LOG What- doesn't exist? Is "TauJets.ptJetSeed"
"TauJets.ptJetSeed", # LOG
"TauClusters.dEta",
"TauClusters.dPhi",
"TauClusters.SECOND_R",
"TauClusters.SECOND_LAMBDA",
"TauClusters.CENTER_LAMBDA",

# 9 vars
"TauJets.centFrac",
"TauJets.etOverPtLeadTrk",
"TauJets.dRmax",
#"TauJets.absipSigLeadTrk",#REMOVED FOR R22
"TauJets.SumPtTrkFrac",
"TauJets.EMPOverTrkSysP",
"TauJets.ptRatioEflowApprox",
"TauJets.mEflowApprox",
"TauJets.ptIntermediateAxis",


# Others
"TauJets.IsTruthMatched",# NOT IN Background files for some dumb reason
#"TauJets.mcEventWeight"
]


variables_dictionary = {"Tracks": ["TauTracks.pt",
                                    "TauTracks.z0sinthetaTJVA",
                                    "TauTracks.dEta",
                                    "TauTracks.dPhi",
                                    "TauTracks.nInnermostPixelHits",
                                    "TauTracks.nPixelHits",
                                    "TauTracks.nSCTHits"],
                        "Clusters": ["TauClusters.et",
                                    "TauClusters.dEta",
                                    "TauClusters.dPhi",
                                    "TauClusters.SECOND_R",
                                    "TauClusters.SECOND_LAMBDA",
                                    "TauClusters.CENTER_LAMBDA"],
                        "Jets": ["TauJets.centFrac",
                                "TauJets.ptJetSeed",
                                "TauJets.etOverPtLeadTrk",
                                "TauJets.dRmax",
                                "TauJets.SumPtTrkFrac",
                                "TauJets.EMPOverTrkSysP",
                                "TauJets.ptRatioEflowApprox",
                                "TauJets.mEflowApprox",
                                "TauJets.ptIntermediateAxis"],
                        "Event": ["TauJets.mcEventWeight"]
                        }

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
"""
variables_dictionary = {"TauTracks": [

                                      "TauTracks.dEta",
                                      "TauTracks.dPhi",
                                      "TauTracks.nInnermostPixelHits",
                                      "TauTracks.nPixelHits",
                                      "TauTracks.nSCTHits",
                                      #"TauTracks.chargedScoreRNN",
                                      #"TauTracks.isolationScoreRNN",
                                      #"TauTracks.conversionScoreRNN",
                                      #"TauTracks.fakeScoreRNN",

                                      "TauTracks.pt",
                                      #"TauTracks.dphiECal",
                                      #"TauTracks.detaECal",
                                      #"TauTracks.jetpt",
                                      "TauTracks.d0TJVA",
                                      "TauTracks.d0SigTJVA",
                                      "TauTracks.z0sinthetaTJVA",
                                      "TauTracks.z0sinthetaSigTJVA"

                                      ],

                        "ConvTrack": ["ConvTrack.dphiECal",
                                      "ConvTrack.dphi",
                                      "ConvTrack.detaECal",
                                      "ConvTrack.deta",
                                      "ConvTrack.pt",
                                      "ConvTrack.jetpt",
                                      "ConvTrack.d0TJVA",
                                      "ConvTrack.d0SigTJVA",
                                      "ConvTrack.z0sinthetaTJVA",
                                      "ConvTrack.z0sinthetaSigTJVA"],

                        "ShotPFO": ["ShotPFO.dphiECal",
                                    "ShotPFO.dphi",
                                    "ShotPFO.detaECal",
                                    "ShotPFO.deta",
                                    "ShotPFO.pt",
                                    "ShotPFO.jetpt"],

                        "NeutralPFO": ["NeutralPFO.dphiECal",
                                       "NeutralPFO.dphi",
                                       "NeutralPFO.detaECal",
                                       "NeutralPFO.deta",
                                       "NeutralPFO.pt",
                                       "NeutralPFO.jetpt",
                                       "NeutralPFO.FIRST_ETA",
                                       "NeutralPFO.SECOND_R",
                                       "NeutralPFO.DELTA_THETA",
                                       "NeutralPFO.CENTER_LAMBDA",
                                       "NeutralPFO.LONGITUDINAL",
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
                                       "NeutralPFO.secondEtaWRTClusterPosition_EM2"],

                        "TauJets": ["TauJets.centFrac",
                                    "TauJets.etOverPtLeadTrk",
                                    "TauJets.dRmax",
                                    "TauJets.SumPtTrkFrac",
                                    "TauJets.ptRatioEflowApprox",
                                    "TauJets.ptIntermediateAxis",
                                    "TauJets.ptJetSeed",
                                    "TauJets.etaJetSeed",
                                    "TauJets.phiJetSeed",
                                    ],

                        "DecayMode": ["TauJets.truthDecayMode"],
                        "Prong": ["TauJets.truthProng"],
                        "Weight": ["TauJets.jet_pt"]}#"TauJets.mcEventWeight"




log_list = [


              "TauTracks.pt",
              "TauTracks.z0sinthetaTJVA",
              "TauTracks.z0sinthetaSigTJVA",
              "ConvTrack.pt",
              "ConvTrack.jetpt",
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
            "TauJets.etaJetSeed",
            "TauJets.phiJetSeed"]














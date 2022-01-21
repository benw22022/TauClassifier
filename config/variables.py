"""
Variables
_____________________________________________________________
Imput Variables configuration
"""

from dataclasses import dataclass
from typing import List
import math
import numpy as np

@dataclass
class Variable:
    """
    A dataclass for handling input variables
    members:
        type (str): The type of variable e.g. TauTracks, NeutralPFO ect...
        name (str): Name of the variable
        max_val (float, optional: default=None): Maximum value variable is allowed to take
        minx_val (float, optional: default=None): Minimum value variable is allowed to take
        lognorm (bool, optional: default=False): If True will take lognorm10 of variable
    """
    
    type: str
    name:str
    max_val: float = None
    min_val: float = None
    lognorm: bool = False
    norm: bool = False

    def standardise(self, data_arr, dummy_val=-1):
        
        if self.min_val is not None:
            data_arr = np.where(data_arr >= self.min_val, data_arr, dummy_val)
        if self.max_val is not None:
            data_arr = np.where(data_arr <= self.max_val, data_arr, dummy_val)

        if self.lognorm:
            data_arr = np.ma.log10(data_arr)
            # data_arr = (data_arr - np.amin(data_arr)) / (np.amax(data_arr) - np.amin(data_arr))
            data_arr = (data_arr - 0) / (math.log10(self.max_val) - 0)
            data_arr = data_arr.filled(dummy_val)
        
        if self.norm:
            data_arr = (data_arr - 0) / (math.log10(self.max_val) - 0)

        return data_arr

    def __call__(self, x):
        return self.name

    def __str__(self):
        return self.name

@dataclass
class VariableHandler:
    """
    A helper class to store and handle input variables
    members:
        variables (List(Variable)): A list of Variable dataclass instances
    """
    variables: List[Variable]

    def add_variable(self, variable):
        """
        Adds a new variable to the VariableHandler
        args:
            variable (Variable): An instance of the Variable dataclass
        returns:
            None
        """
        self.variables.append(variable)

    def get(self, var_type):
        """
        Returns a list of all variables sharing the same type 
        args:
            type (str): Variable type e.g. TauJets, NeutralPFO etc..
        returns:
            (list): A list of Variables sharing the same var_type
        """

        return [variable for variable in self.variables if variable.type == var_type]

    def list(self):
        return [variable.name for variable in self.variables]

    def __len__(self):
        return len(self.variables)

    def __getitem__(self, idx):
        return self.variables[idx]

variable_handler = VariableHandler([])

variable_handler.add_variable(Variable("TauTracks", "TauTracks.nInnermostPixelHits", min_val=0, max_val=3, norm=True))
variable_handler.add_variable(Variable("TauTracks", "TauTracks.nPixelHits", min_val=0, max_val=11, norm=True))
variable_handler.add_variable(Variable("TauTracks", "TauTracks.nSCTHits", min_val=0, max_val=21, norm=True))
variable_handler.add_variable(Variable("TauTracks", "TauTracks.chargedScoreRNN", min_val=0, max_val=1))
variable_handler.add_variable(Variable("TauTracks", "TauTracks.isolationScoreRNN", min_val=0, max_val=1))
variable_handler.add_variable(Variable("TauTracks", "TauTracks.conversionScoreRNN", min_val=0, max_val=1))
variable_handler.add_variable(Variable("TauTracks", "TauTracks.pt", min_val=0, max_val=0.25e7, lognorm=True))
variable_handler.add_variable(Variable("TauTracks", "TauTracks.dphiECal", min_val=0, max_val=1))
variable_handler.add_variable(Variable("TauTracks", "TauTracks.detaECal", min_val=0, max_val=1))
variable_handler.add_variable(Variable("TauTracks", "TauTracks.jetpt", min_val=0, max_val=3e7, lognorm=True))
variable_handler.add_variable(Variable("TauTracks", "TauTracks.d0TJVA", min_val=0, max_val=100, lognorm=True))
variable_handler.add_variable(Variable("TauTracks", "TauTracks.d0SigTJVA", min_val=0, max_val=250, lognorm=True))
variable_handler.add_variable(Variable("TauTracks", "TauTracks.z0sinthetaTJVA", min_val=0, max_val=150, lognorm=True))
variable_handler.add_variable(Variable("TauTracks", "TauTracks.z0sinthetaSigTJVA", min_val=0, max_val=2000, lognorm=True))

variable_handler.add_variable(Variable("ConvTrack", "ConvTrack.dphiECal", min_val=0, max_val=1))
variable_handler.add_variable(Variable("ConvTrack", "ConvTrack.dphi", min_val=0, max_val=1))
variable_handler.add_variable(Variable("ConvTrack", "ConvTrack.detaECal", min_val=0, max_val=1))
variable_handler.add_variable(Variable("ConvTrack", "ConvTrack.deta", min_val=0, max_val=1,))
variable_handler.add_variable(Variable("ConvTrack", "ConvTrack.pt", min_val=0, max_val=5e7, lognorm=True))
variable_handler.add_variable(Variable("ConvTrack", "ConvTrack.jetpt", min_val=0, max_val=3e7, lognorm=True))
variable_handler.add_variable(Variable("ConvTrack", "ConvTrack.d0TJVA", min_val=0, max_val=100, lognorm=True))
variable_handler.add_variable(Variable("ConvTrack", "ConvTrack.d0SigTJVA", min_val=0, max_val=250, lognorm=True))
variable_handler.add_variable(Variable("ConvTrack", "ConvTrack.z0sinthetaTJVA", min_val=0, max_val=100, lognorm=True))
variable_handler.add_variable(Variable("ConvTrack", "ConvTrack.z0sinthetaSigTJVA", min_val=0, max_val=100, lognorm=True))

variable_handler.add_variable(Variable("ShotPFO", "ShotPFO.dphiECal", min_val=0, max_val=1))
variable_handler.add_variable(Variable("ShotPFO", "ShotPFO.dphi", min_val=0, max_val=1))
variable_handler.add_variable(Variable("ShotPFO", "ShotPFO.detaECal", min_val=0, max_val=1))
variable_handler.add_variable(Variable("ShotPFO", "ShotPFO.deta", min_val=0, max_val=1))
variable_handler.add_variable(Variable("ShotPFO", "ShotPFO.pt", min_val=0,  max_val=50000, lognorm=True))
variable_handler.add_variable(Variable("ShotPFO", "ShotPFO.jetpt", min_val=0, max_val=3e7, lognorm=True))

variable_handler.add_variable(Variable("NeutralPFO", "NeutralPFO.dphiECal", min_val=0, max_val=1))
variable_handler.add_variable(Variable("NeutralPFO", "NeutralPFO.dphi", min_val=0, max_val=1))
variable_handler.add_variable(Variable("NeutralPFO", "NeutralPFO.detaECal", min_val=0, max_val=1))
variable_handler.add_variable(Variable("NeutralPFO", "NeutralPFO.deta", min_val=0, max_val=1))
variable_handler.add_variable(Variable("NeutralPFO", "NeutralPFO.pt", min_val=0, max_val=0.5e7, lognorm=True))
variable_handler.add_variable(Variable("NeutralPFO", "NeutralPFO.jetpt", min_val=0, max_val=3e7, lognorm=True))
variable_handler.add_variable(Variable("NeutralPFO", "NeutralPFO.FIRST_ETA", min_val=0, max_val=4, norm=True))
variable_handler.add_variable(Variable("NeutralPFO", "NeutralPFO.SECOND_R", min_val=0, max_val=50000, lognorm=True))
variable_handler.add_variable(Variable("NeutralPFO", "NeutralPFO.DELTA_THETA", min_val=0, max_val=1))
variable_handler.add_variable(Variable("NeutralPFO", "NeutralPFO.CENTER_LAMBDA", min_val=0, max_val=1300, lognorm=True))
variable_handler.add_variable(Variable("NeutralPFO", "NeutralPFO.LONGITUDINAL", min_val=0, max_val=1))
variable_handler.add_variable(Variable("NeutralPFO", "NeutralPFO.SECOND_ENG_DENS", min_val=0, max_val=10, lognorm=True))
variable_handler.add_variable(Variable("NeutralPFO", "NeutralPFO.ENG_FRAC_CORE", min_val=0, max_val=1))
variable_handler.add_variable(Variable("NeutralPFO", "NeutralPFO.NPosECells_EM1", min_val=0, max_val=300, lognorm=True))
variable_handler.add_variable(Variable("NeutralPFO", "NeutralPFO.NPosECells_EM2", min_val=0, max_val=300, lognorm=True))
variable_handler.add_variable(Variable("NeutralPFO", "NeutralPFO.energy_EM1", min_val=0, max_val=0.2e7, lognorm=True))
variable_handler.add_variable(Variable("NeutralPFO", "NeutralPFO.energy_EM2", min_val=0, max_val=0.2e7, lognorm=True))
variable_handler.add_variable(Variable("NeutralPFO", "NeutralPFO.EM1CoreFrac", min_val=0, max_val=1))
variable_handler.add_variable(Variable("NeutralPFO", "NeutralPFO.firstEtaWRTClusterPosition_EM1", min_val=0, max_val=0.25,  lognorm=True))
variable_handler.add_variable(Variable("NeutralPFO", "NeutralPFO.firstEtaWRTClusterPosition_EM2", min_val=0, max_val=0.25,  lognorm=True))
variable_handler.add_variable(Variable("NeutralPFO", "NeutralPFO.secondEtaWRTClusterPosition_EM1", min_val=0, max_val=0.01, lognorm=True))
variable_handler.add_variable(Variable("NeutralPFO", "NeutralPFO.secondEtaWRTClusterPosition_EM2", min_val=0, max_val=0.01, lognorm=True))

variable_handler.add_variable(Variable("TauJets", "TauJets.centFrac", min_val=0, max_val=1.5, norm=True))
variable_handler.add_variable(Variable("TauJets", "TauJets.etOverPtLeadTrk", min_val=0, max_val=30, lognorm=True))
variable_handler.add_variable(Variable("TauJets", "TauJets.dRmax", min_val=0, max_val=1))
variable_handler.add_variable(Variable("TauJets", "TauJets.SumPtTrkFrac", min_val=0, max_val=1))
variable_handler.add_variable(Variable("TauJets", "TauJets.ptRatioEflowApprox", min_val=0, max_val=5, norm=True))
variable_handler.add_variable(Variable("TauJets", "TauJets.mEflowApprox", min_val=0, max_val=0.3e7, lognorm=True))
variable_handler.add_variable(Variable("TauJets", "TauJets.ptJetSeed", min_val=0, max_val=3.5e7, lognorm=True))
variable_handler.add_variable(Variable("TauJets", "TauJets.etaJetSeed", min_val=0, max_val=3, norm=True))
variable_handler.add_variable(Variable("TauJets", "TauJets.phiJetSeed", min_val=0, max_val=3.2, norm=True))

variable_handler.add_variable(Variable("DecayMode", "TauJets.truthDecayMode"))
variable_handler.add_variable(Variable("Prong", "TauJets.truthProng"))
variable_handler.add_variable(Variable("Weight", "TauJets.ptJetSeed"))
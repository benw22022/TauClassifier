"""
Attempt to make a more versatile dataset
"""

import os
import glob
import uproot
import tensorflow as tf
import awkward as ak
from config.config import ntuple_dir
import tensorflow_io.arrow as arrow_io
from pyarrow.feather import write_feather
import pyarrow.parquet as pq
import pyarrow as pa
import tensorflow_io as tfio

var_list = {
"TauJets.mcEventNumber": tf.int32,
"TauJets.mcEventWeight": tf.float32,
"TauJets.truthDecayMode": tf.int32,
"TauJets.nVtxPU": tf.float32,
"TauJets.jet_eta": tf.float32,
"TauJets.jet_phi": tf.float32,
"TauJets.jet_pt": tf.float32,
"TauJets.nTracks": tf.int32,
"TauTracks.pt": tf.float32,
"TauTracks.jetpt": tf.float32,
"TauTracks.dphi": tf.float32,
"TauTracks.dphiECal": tf.float32,
"TauTracks.deta": tf.float32,
"TauTracks.detaECal": tf.float32,
"NeutralPFO.pt": tf.float32,
"NeutralPFO.jetpt": tf.float32,
"NeutralPFO.dphi": tf.float32,
"NeutralPFO.dphiECal": tf.float32,
"NeutralPFO.deta": tf.float32,
"NeutralPFO.detaECal": tf.float32,
"ShotPFO.pt": tf.float32,
"ShotPFO.jetpt": tf.float32,
"ShotPFO.dphi": tf.float32,
"ShotPFO.dphiECal": tf.float32,
"ShotPFO.deta": tf.float32,
"ShotPFO.detaECal": tf.float32,
"ConvTrack.pt": tf.float32,
"ConvTrack.jetpt": tf.float32,
"ConvTrack.dphi": tf.float32,
"ConvTrack.dphiECal": tf.float32,
"ConvTrack.deta": tf.float32,
"ConvTrack.detaECal": tf.float32,
"ConvTrack.d0TJVA": tf.float32,
"ConvTrack.d0SigTJVA": tf.float32,
"ConvTrack.z0sinthetaTJVA": tf.float32,
"ConvTrack.z0sinthetaSigTJVA": tf.float32,
"NeutralPFO.pi0BDT": tf.float32,
"NeutralPFO.ptSubRatio": tf.float32,
"NeutralPFO.NHitsInEM1": tf.int32,
"NeutralPFO.secondEtaWRTClusterPosition_EM1": tf.float32,
"NeutralPFO.energyfrac_EM2": tf.float32,
"NeutralPFO.FIRST_ETA": tf.float32,
"NeutralPFO.SECOND_R": tf.float32,
"NeutralPFO.DELTA_THETA": tf.float32,
"NeutralPFO.CENTER_LAMBDA": tf.float32,
"NeutralPFO.LONGITUDINAL": tf.float32,
"NeutralPFO.ENG_FRAC_EM": tf.float32,
"NeutralPFO.ENG_FRAC_CORE": tf.float32,
"NeutralPFO.SECOND_ENG_DENS": tf.float32,
"NeutralPFO.EM1CoreFrac": tf.float32,
"NeutralPFO.NPosECells_EM1": tf.float32,
"NeutralPFO.NPosECells_EM2": tf.float32,
"NeutralPFO.firstEtaWRTClusterPosition_EM1": tf.float32,
"NeutralPFO.secondEtaWRTClusterPosition_EM2": tf.float32,
"NeutralPFO.LATERAL": tf.float32,
"NeutralPFO.ENG_FRAC_MAX": tf.float32,
"NeutralPFO.NPosECells_PS": tf.float32,
"NeutralPFO.firstEtaWRTClusterPosition_EM1": tf.float32,
"NeutralPFO.energy_EM1": tf.float32,
"NeutralPFO.energy_EM2": tf.float32,
"ShotPFO.nCellsInEta": tf.float32,
"ShotPFO.pt1": tf.float32,
"ShotPFO.pt3": tf.float32,
"ShotPFO.pt5": tf.float32,
"ShotPFO.nPhotons": tf.int32,
"TauTracks.charge": tf.float32,
"TauTracks.pt": tf.float32,
"TauTracks.eta": tf.float32,
"TauTracks.phi": tf.float32,
"TauTracks.chiSquared": tf.float32,
"TauTracks.nDoF": tf.int32,
"TauTracks.nInnermostPixelHits": tf.int32,
"TauTracks.nPixelHits": tf.int32,
"TauTracks.nSCTHits": tf.int32,
"TauTracks.z0sinthetaTJVA": tf.float32,
"TauTracks.z0sinthetaSigTJVA": tf.float32,
"TauTracks.d0TJVA": tf.float32,
"TauTracks.d0SigTJVA": tf.float32,
"TauTracks.chargedScoreRNN": tf.float32,
"TauTracks.isolationScoreRNN": tf.float32,
"TauTracks.conversionScoreRNN": tf.float32,
"TauTracks.fakeScoreRNN": tf.float32,
"TauJets.nTracks": tf.int32,
"TauJets.nTracksFiltered": tf.int32,
"TauJets.ptJetSeed": tf.float32,
"TauJets.etaJetSeed": tf.float32,
"TauJets.phiJetSeed": tf.float32,
"TauJets.centFrac": tf.float32,
"TauJets.EMPOverTrkSysP": tf.float32,
"TauJets.innerTrkAvgDist": tf.float32,
"TauJets.ptRatioEflowApprox": tf.float32,
"TauJets.dRmax": tf.float32,
"TauJets.trFlightPathSig": tf.float32,
"TauJets.mEflowApprox": tf.float32,
"TauJets.SumPtTrkFrac": tf.float32,
"TauJets.absipSigLeadTrk": tf.float32,
"TauJets.massTrkSys": tf.float32,
"TauJets.etOverPtLeadTrk": tf.float32,
"TauJets.ptIntermediateAxis": tf.float32,
}

def save_dataset(files, is_jet=False):

    for batch in uproot.iterate(files, filter_name=list(var_list.keys()), library='ak'):

        print(len(batch))

        if is_jet: 
            batch["TauJets.truthDecayMode"] = 0
        else:
            batch["TauJets.truthDecayMode"] = batch["TauJets.truthDecayMode"] + 1

        ak.to_parquet(batch, f"experimental/data.parquet")
        ak.to_parquet.dataset("experimental")
        
        # arrow_arrs = [ak.to_arrow(batch[v]) for v in list(var_list.keys())]

        # print(len(arrow_arrs))
        # print(len(var_list))

        # pa_batch = pa.RecordBatch.from_arrays(arrow_arrs, list(var_list.keys()))
        # pa_table = pa.Table.from_batches([pa_batch])

        # write_feather(pa_table, 'experimental/data.feather')

        # df = tfio.arrow.ArrowFeatherDataset(ak.to_arrow(batch), list(var_list.keys()), list(var_list.values()))
        # ds = tfio.experimental.IODataset.from_parquet("experimental/data.parquet")
        
        # for i, row in enumerate(ds.take(1)):
        #     print(row)

        break


    # df = tfio.arrow.ArrowFeatherDataset('experimental/data.feather', list(var_list.keys()), list(var_list.values()))



if __name__ == "__main__":

    tau_files = glob.glob(os.path.join(ntuple_dir, "*Gammatautau*", "*.root"))[0]
    jet_files = glob.glob(os.path.join(ntuple_dir, "*JZ*", "*.root"))[0]

    save_dataset(tau_files)

    ds = tfio.experimental.IODataset.from_parquet("experimental/data.parquet")
    # ds = ak.from_parquet("experimental/data.parquet")

    for i, row in enumerate(ds):
        for key in row:
            print(f"{key} -- {row[key]}")
        # print(row[b'"TauTracks.deta.list.item'])
        break
        # print(row[b'TauTracks.z0sinthetaTJVA.list.item'])




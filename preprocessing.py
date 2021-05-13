"""
This is where the data pre-processing will go
"""

import numpy as np
from keras import backend as K
import uproot
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.optimize import curve_fit
import math

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
	return  np.log(m, out=np.zeros_like(m), where=(m > 1))


def pt_reweight(pt):
	# Hardcode coeffs for now for convenience- TODO: make this an automated routine
	result = 4.486717192254409 * np.exp(12.230191960759665* pt/1e6) + 0.14197785024185136
	result = np.nan_to_num(result, posinf=0, neginf=0, copy=False)  # Make sure nothing dodgy happens!
	result[result > 4] = 4  # Clip weights to a maximum value of 5 to prevent abnormally large weights
	return result


def reweight_func(x, a, b, c):
	return a * np.exp(-b * x) + c


def get_r_squared(y, y_fit):
	# residual sum of squares
	ss_res = np.sum((y - y_fit) ** 2)

	# total sum of squares
	ss_tot = np.sum((y - np.mean(y)) ** 2)

	# r-squared
	r2 = 1 - (ss_res / ss_tot)

	return r2


def get_bin_centres(bin_edges):
	bin_centres = np.zeros(len(bin_edges) - 1)
	for i in range(1, len(bin_edges)):
		bin_centres[i - 1] = bin_edges[i] + (bin_edges[i] - bin_edges[i - 1]) / 2
	return bin_centres


if __name__ == "__main__":
	ntuple_dir = "E:\\NTuples\\TauClassifier"
	sig_files = glob.glob(f"{ntuple_dir}\\*Gammatautau*\\*.root")
	bkg_files = glob.glob(f"{ntuple_dir}\\*JZ*\\*.root")

	cuts = "(TauJets.jet_pt > 15000.0) & (TauJets.jet_pt < 10000000.0)"

	sig_pt = uproot.concatenate(sig_files, filter_name="TauJets.jet_pt",
								cut=cuts, library='np')

	bkg_pt = uproot.concatenate(bkg_files, filter_name="TauJets.jet_pt", cut=cuts,
								library='np')

	bkg_pt = bkg_pt["TauJets.jet_pt"]
	bkg_pt = np.sort(bkg_pt) / 1e6

	sig_pt = sig_pt["TauJets.jet_pt"]
	sig_pt = np.sort(sig_pt) / 1e6

	# Binning
	bin_edges = np.percentile(bkg_pt, np.linspace(0.0, 100.0, 50))

	# Reweighting coefficient
	sig_hist, sig_freq = np.histogram(sig_pt, bins=bin_edges, density=True)
	bkg_hist, bkg_freq = np.histogram(bkg_pt, bins=bin_edges, density=True)
	coeff = sig_hist / bkg_hist

	# Fit the histogram
	bin_centres = get_bin_centres(bin_edges)
	p_opt, _ = curve_fit(reweight_func, bin_centres, coeff, p0=(1, 1, 0.4))

	# Compute goodness of fit using Coefficient of Determination
	r_squared = get_r_squared(coeff, reweight_func(bin_centres, *p_opt))
	print("Fitted function y = a exp(b x) + c")
	print(f"coefficients: a = {p_opt[0]}  b = {p_opt[1]}  c = {p_opt[2]}")
	print(f"r-squared = {r_squared}")

	# Plot histogram
	fig, ax = plt.subplots()
	ax.scatter(bin_centres, coeff, label="data")
	ax.plot(bin_centres, reweight_func(bin_centres, *p_opt), label="fit", color="orange")

	tex = r'$y = ae^{-bx} + c$'
	tex += "\na = {0:.3g}".format(p_opt[0])
	tex += "\nb = {0:.3g}".format(p_opt[1])
	tex += "\nc = {0:.3g}".format(p_opt[2])
	tex += '\n$\mathcal{R}^{2}$'
	tex += " = {0:.3g}".format(r_squared)
	ax.text(4.05, 2.95, tex, fontsize=11, va='top')

	ax.set_xlabel("pT MeV")
	ax.set_ylabel("coeff")
	ax.legend()
	plt.show()
	plt.savefig("pt_reweight.png")
"""
Plot Variables
_____________________________________________
Script to plot variables
TODO: Options for plotting tau-jets, tau decay mode ect... Can't just comment the bits out
"""

import uproot
import glob
import os
import numpy as np
import awkward as ak
from tqdm import tqdm
import matplotlib.pyplot as plt
from config.config import get_cuts
from config.config import ntuple_dir
from source.utils import logger
from config.variables import variable_handler


class Plotter:

    def __init__(self, file_list, name, colour='blue', reweighter=None, cuts=""):
        self.file_list = file_list
        self.cuts = cuts
        self.reweighter = reweighter
        self.label = name
        self.colour = colour

    def plot(self, quantity, ax, bins=50):
        data = uproot.concatenate(self.file_list, filter_name=quantity.name, cut=self.cuts, library='ak')
        data = data[quantity.name]
        if "TauJets" not in quantity.name:
            data = ak.pad_none(data, 10, clip=True, axis=1)
        
        data = ak.to_numpy(abs(data))
        try:
            data = data.filled(-1)
        except AttributeError:
            pass
        data = quantity.standardise(data).ravel()
        
        ax.hist(data, bins=np.linspace(-1e-12, np.amax(data), 50), density=True, histtype='step', label=self.label, color=self.colour)

def plot_variable(plotters, quantity):
    """
    Plots a histogram of a variable
    :param plotters (list[Plotter]): A list of Plotter classes
    :param quantity (str): The variable to be plotted
    :returns None:
    """

    fig, ax = plt.subplots()
    for plotter in plotters:
        plotter.plot(quantity, ax)
    ax.legend()
    ax.set_ylabel("# Events Unit Normalised")
    ax.set_xlabel(quantity)
    # ax.set_yscale('log')
    plt.savefig(os.path.join("plots", "variables", f"{quantity}.png"))
    plt.close()

def plot_variables():
    """
    Function to plot all variables in the variables_dictionary
    """

    cuts = get_cuts()
    jet_files = glob.glob(os.path.join(f"{ntuple_dir}", "*JZ*/*.root"))
    tau_files = glob.glob(os.path.join(f"{ntuple_dir}", "*Gammatautau*/*.root"))

    jet_plotter = Plotter(jet_files[0], "Jets", cuts=cuts["JZ1"], colour='blue')
    tau_plotter = Plotter(tau_files[0], "Taus", cuts=cuts["Gammatautau"], colour='orange')

    plotter_1p0n = Plotter(tau_files[0], "1p0n", cuts=get_cuts(decay_mode=0)["Gammatautau"], colour='red')
    plotter_1p1n = Plotter(tau_files[0], "1p1n", cuts=get_cuts(decay_mode=1)["Gammatautau"], colour='orange')
    plotter_1pXn = Plotter(tau_files[0], "1pXn", cuts=get_cuts(decay_mode=2)["Gammatautau"], colour='green')
    plotter_3p0n = Plotter(tau_files[0], "3p0n", cuts=get_cuts(decay_mode=3)["Gammatautau"], colour='cyan')
    plotter_3pXn = Plotter(tau_files[0], "3pXn", cuts=get_cuts(decay_mode=4)["Gammatautau"], colour='magenta')

    all_plotters = (jet_plotter, plotter_1p0n, plotter_1p1n, plotter_1pXn, plotter_3p0n, plotter_3pXn)

    os.system("rm plots/variables/*.png")

    # logger.log(f"Plotting {len(variable_handler)} histograms comparing taus and jets")
    # for variable in tqdm(variable_handler):
    #     plot_variable((tau_plotter, jet_plotter), variable)

    logger.log(f"Plotting {len(variable_handler)}  histograms comparing tau decay modes")
    for variable in tqdm(variable_handler):
        plot_variable(all_plotters, variable)

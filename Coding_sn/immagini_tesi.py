import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tvb.simulator.lab import *
from scipy.spatial import KDTree
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd

def plot_metrics(
        conv_met = pd.read_csv(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Coding\Risultati_03_02\nsga2_convergence_metrics_03_02.csv")
        ):
    
    gen = conv_met['generation']
    plt.figure(figsize=(8, 5))
    plt.plot(gen, conv_met["hypervolume"], marker="o")

    plt.xlabel("Generation", fontsize=26, labelpad=15)
    plt.ylabel("Hypervolume", fontsize=26, labelpad=15)
    plt.title("Hypervolume convergence", fontsize=26, pad=20)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    ax = plt.gca()

    # Rimuovi bordi destro e superiore
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Ticks
    ax.tick_params(axis='both', which='major',
                direction='out', length=8, width=2,
                labelsize=26, colors='black', pad=10)

    # Numero di ticks (aggiusta nbins a piacere)
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))

    plt.tight_layout()
    plt.savefig(r"C:\Users\User\OneDrive - University of Pisa\Desktop\Thesis\Immagini_tesi\Results\risultati_hypervolume2.png", dpi=250)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(gen, conv_met["spacing"], marker="o")

    plt.xlabel("Generation", fontsize=26, labelpad=15)
    plt.ylabel("Spacing", fontsize=26, labelpad=15)
    plt.title("Spacing convergence", fontsize=26, pad=20)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    ax = plt.gca()

    # Rimuovi bordi destro e superiore
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Ticks
    ax.tick_params(axis='both', which='major',
                direction='out', length=8, width=2,
                labelsize=26, colors='black', pad=10)

    # Numero di ticks (aggiusta nbins a piacere)
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))

    plt.tight_layout()
    plt.savefig(r"C:\Users\User\OneDrive - University of Pisa\Desktop\Thesis\Immagini_tesi\Results\risultati_spacing2.png", dpi=250)
    plt.close()

plot_metrics()
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import r2_score, mean_squared_error
import time
import os
import matplotlib.pyplot as plt

# pymoo
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from pymoo.core.termination import Termination

# tue funzioni
from utils import preprocess, new_parcellation_simulate
from data_analysis_utils import real_vs_simulated_FC, real_vs_simulated_PSD
import global_variable_creation as gv

import numpy as np
import argparse
from tvb.simulator.lab import *
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils import skin_and_sensors, brain_activity, simulate, preprocess, eeg_plot, simulate_different_combination_parameters, simulate_parametri_lorenzo
from utils import functional_connectivity, psd, functional_connectivity_band, functional_connectivity_img_coh, psd_simil_lorenzo, Simulate2, functional_connectivity_lorenzo
from utils import make_train, simulate_with_stimulus, stimulus_visualization, new_parcellation_simulate, deterministic_new_parcellation_simulate
from classes import MultiStimuliRegion
from data_analysis_utils import real_vs_simulated_FC, real_vs_simulated_PSD, NSGAII, analyze_regular_data, analyze_reduced_data, load_simulation_results
import global_variable_creation as gv

from joblib import Parallel, delayed

def check_majority_seeds(df, ratio_threshold=0.5):
    """
    For every couple (sf, channel), we check if at least threshold of
    seeds satisfies is_greater condition 
    
    Returns:
        Pandas DataFrame with columns: sf, channel, fraction_greater, majority_satisfied
    """
    result = (
        df.groupby(["sf", "channel"])["is_greater"]
        .mean()
        .reset_index()
        .rename(columns={"is_greater": "fraction_greater"})
    )
    result["majority_satisfied"] = result["fraction_greater"] >= ratio_threshold
    return result

def best_sf(df, ratio_threshold=0.5):
    """
    Returns value of sf for which the highes number of channels satisfy
    is greater condition at least for threshold*seeds number.
    """
    majority_result = check_majority_seeds(df, ratio_threshold)
    
    channels_per_sf = (
        majority_result
        .groupby("sf")["majority_satisfied"]
        .sum()
        .reset_index()
        .rename(columns={"majority_satisfied": "n_channels_satisfied"})
    )
    
    best = channels_per_sf.loc[channels_per_sf["n_channels_satisfied"].idxmax()]
    return best

def visualize_stimulus_amplitude(df, channel=0, seed=0):
    filtered_seed_channel = df.query("channel == @channel and seed == @seed")
    scale_factors = filtered_seed_channel["sf"]
    pre_stimulus_amplitudes = filtered_seed_channel["amplitude_pre_stimulus"]
    post_stimulus_amplitudes = filtered_seed_channel["amplitude_post_stimulus"]

    plt.semilogx(scale_factors, post_stimulus_amplitudes, label='post_stim')
    plt.semilogx(scale_factors, pre_stimulus_amplitudes, label='pre_stim')
    #plt.yticks(np.linspace(0,25000, 26))
    plt.legend()
    plt.ylabel('amplitude')
    plt.xlabel('weights multiplication factor')
    plt.title(f"Post stimulus amplitude vs stimulus intensity channel {channel}, seed{seed}")
    plt.grid()
    plt.show()

def main():

    #data= np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Coding_sn\tavg_protocollo_1.npy")
    """
    columns = [
        "baseline",
        "amplitude_pre_stimulus",
        "amplitude_post_stimulus",
        "sf",
        "seed",
        "channel"
    ]

    df = pd.DataFrame(data, columns=columns)
    """
    df= pd.read_pickle(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Tms_results_3\tms_seeds_a0_t0_f0.5_s0")
    visualize_stimulus_amplitude(df=df)

    amplitude_threshold = 1.5
    ratio_threshold = 0.5
    print_sf = 38
    psdp = print_sf+1
    psdm = print_sf-1

    df["is_greater"] = df["amplitude_post_stimulus"] > (amplitude_threshold * df["amplitude_pre_stimulus"])

    majority_result = check_majority_seeds(df, ratio_threshold=ratio_threshold)
    best_majority=majority_result.query("sf >= @psdm and sf <= @psdp")
    best_majority["majority_satisfied"].to_csv("best_majority.csv", index=False)
    np.save("best_majority.npy", best_majority["majority_satisfied"].to_numpy())
    #result = df.groupby(["sf", "channel"])["is_greater"].mean()
    #result_reset = result.reset_index()

    #print(result_reset.query("sf >= 100"))

    best = best_sf(df, ratio_threshold=ratio_threshold)
    print(f"Best sf: {best['sf']} with {int(best['n_channels_satisfied'])} channels over threshold")

    
if __name__ == "__main__":
    main()
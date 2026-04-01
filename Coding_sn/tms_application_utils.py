"""
Visualization, simulation, and signal processing utilities for brain activity modeling
--------------------------------------------------------------------------------------

This module provides a set of helper functions for:
- Visualizing brain surfaces, sensors, and connectivity matrices
- Simulating brain activity using the Jansen–Rit model within The Virtual Brain (TVB)
- Preprocessing EEG and region-based signals
- Computing and plotting functional connectivity and power spectral density (PSD)
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import r2_score, mean_squared_error
import tvb.simulator.lab as tvb
from tvb.simulator.lab import equations,patterns, monitors
from tvb.simulator.lab import *
from tvb.datatypes import graph
from tvb.datatypes.time_series import TimeSeriesRegion 
from scipy.stats import pearsonr, wasserstein_distance
from scipy.signal import butter, filtfilt, csd, welch
from scipy import signal
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from pymoo.core.termination import Termination
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.projections import  ProjectionSurfaceEEG#,ProjectionMatrix,
from tvb.datatypes.sensors import SensorsEEG
import copy

from classes import JRPSP, Simulator_Modified
from global_variable_creation import get_eeg_monitor, get_conn
import global_variable_creation as gv
from additional_utils import macro_fc

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
import logging

def stimulation_analysis(times, timeseries, stimulus_onset, preprocess_window=1000):
    mask_pre_stimulus = (times < (stimulus_onset - 100))
    mask_post_stimulus = (times >= stimulus_onset)
    assert(stimulus_onset >= preprocess_window + 1000)
    timeseries_mask_pre_stimulus = timeseries[mask_pre_stimulus]
    timeseries_mask_post_stimulus = timeseries[mask_post_stimulus]
    timeseries_baseline = np.mean(timeseries_mask_pre_stimulus)
    pre_stimulus_amplitude = np.max(np.abs(timeseries_mask_pre_stimulus - timeseries_baseline))
    post_stimulus_amplitude = np.max(np.abs(timeseries_mask_post_stimulus - timeseries_baseline))
    return timeseries_baseline, pre_stimulus_amplitude, post_stimulus_amplitude

def tms_pulse_kick_all_channels(scale_fact, tau, abeta, seed, period, weights_vector, stimulus_onset):
    """
    Simulation for given scale_factor and stimulus
    Return: (baseline, pre_amp, post_amp, sf, seed, channel)
    """
    print('abeta', abeta)
    print('tau',tau)
    for name in logging.root.manager.loggerDict:
        if name.startswith('tvb'):
            logging.getLogger(name).setLevel(logging.ERROR)
            logging.getLogger(name).handlers = []

    gv.random_seed = seed
    sim_time = stimulus_onset + 2000 + 500

    train = make_train(
        np.arange(0, 86),
        scale_fact * weights_vector,
        onset=stimulus_onset,
        T=period,
        tau=1.0
    )

    stimulus = MultiStimuliRegion(train)
    stimulus.configure_space()
    time = np.r_[stimulus_onset - 500:stimulus_onset + 500 + 2000:0.5]
    stimulus.configure_time(time)
    pattern = stimulus()


    ttavg_stim, tavg_healthy_stim, teeg_stim, eeg_healthy_stim = new_parcellation_simulate(
        g=gv.coup,
        velocity=gv.vel,
        noise=gv.noi,
        a=gv.aa,
        b=gv.bb,
        sim_time=sim_time,
        stimulus=stimulus,
        tau_stage=tau,
        abeta_stage=abeta
    )

    ttavg_stim = ttavg_stim[1000:]
    tavg_healthy_stim = tavg_healthy_stim[1000:, :, :, :]
    teeg_stim = teeg_stim[1000:]
    eeg_healthy_stim = eeg_healthy_stim[1000:, :, :, :]

    signal = tavg_healthy_stim
    time_array = ttavg_stim

    # array shape: (time, 1, channels, 1)
    n_channels = signal.shape[2]

    rows = []
    for channel in range(n_channels):
        baseline, pre_amp, post_amp = stimulation_analysis(
            times=time_array,
            timeseries=signal[:, 0, channel, 0],
            stimulus_onset=stimulus_onset
        )
        rows.append((baseline, pre_amp, post_amp, scale_fact, gv.random_seed, channel))

    return rows

def tms_seeds_analysis(file_name, tau, abeta, period, weights_vector, stimulus_onset, tested_values=100, n_seeds=20):

    scale_factor_span = np.logspace(-1, 2.5, tested_values)#np.logspace(-1, 2.5, tested_values)
    seed_span = np.arange(n_seeds)  # 0 ... 19

    # Genera tutte le coppie (sf, seed)
    pairs = [(sf, seed) for sf in scale_factor_span for seed in seed_span]

    # Una simulazione per coppia (sf, seed), parallelizzata
    all_results = Parallel(n_jobs=10)(
        delayed(tms_pulse_kick_all_channels)(
            stimulus_onset=stimulus_onset,
            period=period,
            tau=tau,
            abeta=abeta,
            weights_vector=weights_vector,
            scale_fact=sf,
            seed=seed
        )
        for sf, seed in pairs
    )

    # Appiattisci: lista di liste → lista di righe
    rows = [row for sim_rows in all_results for row in sim_rows]

    # Costruisci DataFrame e salva
    df = pd.DataFrame(rows, columns=["baseline", "amplitude_pre_stimulus", "amplitude_post_stimulus", "sf", "seed", "channel"])
    df["channel"] = df["channel"].astype(int)
    df["seed"] = df["seed"].astype(int)

    #df.to_csv("tavg_protocollo_1.csv", index=False)
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    df.to_pickle(file_name)

    #print(f"Completed: {tested_values} sf × {n_seeds} seeds × {df['channel'].nunique()} channels = {len(df)} total rows.")
    return df.to_numpy()


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
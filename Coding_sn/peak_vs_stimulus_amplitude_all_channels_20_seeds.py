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

from joblib import Parallel, delayed

def main():

    weights_vector = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\prova_protocollo_1_stimulus_vector_TVB.npy")

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

    def tms_pulse_kick_all_channels(scale_fact, seed, start_time=2500, period=2000.0, impulse_time_amplitude=1.0):
        """
        One simulation for each couple (scale_factor, seed)
        Return: (baseline, pre_amp, post_amp, sf, seed, channel)
        """
        for name in logging.root.manager.loggerDict:
            if name.startswith('tvb'):
                logging.getLogger(name).setLevel(logging.ERROR)
                logging.getLogger(name).handlers = []

        gv.random_seed = seed

        train = make_train(
            np.arange(0, 86),
            scale_fact * weights_vector,
            onset=start_time,
            T=period,
            tau=impulse_time_amplitude
        )

        stimulus = MultiStimuliRegion(train)
        stimulus.configure_space()
        time = np.r_[start_time - 500:start_time + 500 + period:0.5]
        stimulus.configure_time(time)
        pattern = stimulus()

        sim_time = start_time + period + 500
        ttavg_stim, tavg_healthy_stim, teeg_stim, eeg_healthy_stim = new_parcellation_simulate(
            g=gv.coup,
            velocity=gv.vel,
            noise=gv.noi,
            a=gv.aa,
            b=gv.bb,
            sim_time=sim_time,
            stimulus=stimulus,
            tau_stage=gv.tau_stage,
            abeta_stage=gv.abeta_stage
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
                stimulus_onset=start_time
            )
            """
            if (channel == 0) & (gv.random_seed == 0) & (scale_fact > 5*10^3):
                plt.figure(figsize=(10,12))
                plt.plot(time_array, signal[:, 0, 0, 0])
                plt.plot(time_array, baseline*np.ones(time_array.shape), label='baseline')
                plt.plot(time_array, (baseline+pre_amp)*np.ones(time_array.shape), label='pre_amp', color = 'red')
                plt.plot(time_array, (baseline-pre_amp)*np.ones(time_array.shape), color = 'red')
                plt.plot(time_array, (baseline+post_amp)*np.ones(time_array.shape), label='post_amp', color = 'green')
                plt.plot(time_array, (baseline-post_amp)*np.ones(time_array.shape), label='post_amp', color = 'green')
                plt.title('Channel 0')
                plt.legend()
                plt.show()
            """
            rows.append((baseline, pre_amp, post_amp, scale_fact, seed, channel))



        return rows

    tested_values = 100
    n_seeds = 10
    scale_factor_span = np.logspace(-1, 2.5, tested_values)
    seed_span = np.arange(n_seeds)  # 0 ... 19

    # Genera tutte le coppie (sf, seed)
    pairs = [(sf, seed) for sf in scale_factor_span for seed in seed_span]

    # Una simulazione per coppia (sf, seed), parallelizzata
    all_results = Parallel(n_jobs=10)(
        delayed(tms_pulse_kick_all_channels)(sf, seed) for sf, seed in pairs
    )

    # Appiattisci: lista di liste → lista di righe
    rows = [row for sim_rows in all_results for row in sim_rows]

    # Costruisci DataFrame e salva
    df = pd.DataFrame(rows, columns=["baseline", "amplitude_pre_stimulus", "amplitude_post_stimulus", "sf", "seed", "channel"])
    df["channel"] = df["channel"].astype(int)
    df["seed"] = df["seed"].astype(int)

    df.to_csv("prova_disperata.csv", index=False)
    np.save("prova_disperata.npy", df.to_numpy())

    print(f"Completed: {tested_values} sf × {n_seeds} seeds × {df['channel'].nunique()} channels = {len(df)} total rows.")

if __name__ == "__main__":
    main()
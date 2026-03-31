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

import os

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
import numpy as np

def main():

    weights_vector = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\simnibs_simulation_M1_stimulus_vector_TVB.npy")
    
    def stimulation_analysis(teeg, eeg, stimulus_onset, preprocess_window=1000):
        mask_pre_stimulus = teeg < (stimulus_onset-100)
        mask_post_stimulus = (teeg >= stimulus_onset)
        assert(stimulus_onset >= preprocess_window+1000)
        eeg_mask_pre_stimulus = eeg[mask_pre_stimulus]
        eeg_mask_post_stimulus = eeg[mask_post_stimulus]
        eeg_baseline = np.mean(eeg_mask_pre_stimulus)
        pre_stimulus_amplitude = np.max(np.abs(eeg_mask_pre_stimulus-eeg_baseline))
        post_stimulus_amplitude = np.max(np.abs(eeg_mask_post_stimulus-eeg_baseline))
        return eeg_baseline, pre_stimulus_amplitude, post_stimulus_amplitude

    def tms_pulse_kick(scale_fact, start_time = 2500, period = 2000.0, impulse_time_amplitude = 1.0):
        
        train = make_train(np.arange(0,86), scale_fact*weights_vector, onset=start_time, T=period, tau=impulse_time_amplitude)
        
        stimulus = MultiStimuliRegion(train)
        stimulus.configure_space()
        time = np.r_[start_time-500:start_time+500+period:0.5]
        stimulus.configure_time(time)
        pattern = stimulus()
  
        sim_time = start_time + period - 100
        ttavg_stim, tavg_healthy_stim, teeg_stim, eeg_healthy_stim = new_parcellation_simulate(
            g=gv.coup,
            velocity=gv.vel,
            noise=gv.noi,
            a=gv.aa,
            b=gv.bb,
            sim_time=sim_time,
            stimulus=stimulus,
            tau_stage=gv.tau_stage,
            abeta_stage=gv.abeta_stage)
        
        ttavg_stim = ttavg_stim[1000:]
        tavg_healthy_stim = tavg_healthy_stim[1000:, :, :, :]
        teeg_stim = teeg_stim[1000:]
        eeg_healthy_stim = eeg_healthy_stim[1000:, :, :, :]

        # I will give just channel 0 (Cz) because this is the channel use in the reference article
        eeg_baseline, pre_stimulus_amplitude, post_stimulus_amplitude = stimulation_analysis(
            teeg=teeg_stim, 
            eeg=eeg_healthy_stim[:,0,0,0], 
            stimulus_onset=start_time
            )

        return eeg_baseline, pre_stimulus_amplitude, post_stimulus_amplitude
    
    tested_values = 5
    scale_factor_span = np.logspace(-1,4,tested_values)

    results = Parallel(n_jobs=10)(
        delayed(tms_pulse_kick)(sf) for sf in scale_factor_span
    )

    eeg_features = np.array(results)
    eeg_features = np.column_stack((scale_factor_span, eeg_features))
    np.save("eeg_features.npy", eeg_features)

if __name__ == "__main__":
    main()
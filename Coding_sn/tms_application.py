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

def main():

    print('You are going to perform a tms stimulation over a choosen patient (identified by its random seed) in a choosen stage. ' \
    'You can also select the spot of application (1 mm scalp distance is fixed), the intensity and the frequency of stimulation.' \
    'Intensity can be any semi-positive number, while frequency of stimulation can only be low (default, 0.5 Hz) or high (10 Hz).')

    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=int, help="Tau stage (from 0 to 5)")
    parser.add_argument("--abeta", type=int, help="Beta Amyloid stage (from 0 to 5)")
    parser.add_argument("--rnd", type=int, help="Simulation random seed")
    parser.add_argument("--tms_intensity", type=float, help="Intensity of stimulation")
    parser.add_argument("--high_f", action="store_true", help="High intensity stimulation")
    parser.add_argument("--low_f", action="store_true", help="Low intensity stimulation")
    parser.add_argument("--show_plots", action="store_true", help="Show plots on screen")
    parser.add_argument("--save_plots", action="store_true", help="Save plots to disk")
    parser.add_argument("--save_folder", type=str, help="Folder to save figures and data")
    parser.add_argument("--spot", type=int, help="0 = PCU, 1 = Fp, 2 = T3/T7, 3 = T4/T8")
    args = parser.parse_args()

    # Per ogni parametro: se non passato da CLI, chiedi interattivamente
    if args.tau is None:
        args.tau = int(input("Tau stage (0-5): "))

    if args.abeta is None:
        args.abeta = int(input("Beta Amyloid stage (0-5): "))

    if args.rnd is None:
        args.rnd = int(input("Random seed: "))

    if args.tms_intensity is None:
        args.tms_intensity = float(input("Intensity: "))

    if args.spot is None:
        args.spot = int(input("Spot (0 = PCU, 1 = Fp, 2 = T3/T7, 3 = T4/T8): "))

    if args.high_f and args.low_f:
        raise ValueError('you cannot simulate both high and low intenisty')

    if args.low_f is True:
        gv.high_tms_freq = False

    # high_f è un bool (store_true), quindi il default è già False
    # ma se vuoi chiederlo esplicitamente:
    if (not args.high_f) & (not args.low_f) :
        risposta = input("High intensity? (y/n): ").strip().lower()
        args.high_f = risposta == 'y'

    print(args.spot)

    if (args.spot < 0) | (args.spot > 3):
        raise ValueError('Spot codes range from 0 to 3')

    gv.show_plots = args.show_plots
    gv.save_plots = args.save_plots
    gv.random_seed = args.rnd
    gv.tau_stage = args.tau
    gv.abeta_stage = args.abeta
    gv.tms_intensity = args.tms_intensity
    gv.high_tms_freq = args.high_f
    gv.stimulation_spot = args.spot

    if args.save_folder:
        gv.fig_folder = os.path.join(gv.fig_folder, args.save_folder)
        gv.data_folder = os.path.join(gv.data_folder, args.save_folder)
        if len(gv.fig_folder) > 160:
            raise AttributeError(
                "file directory has to be shorter."
            )

    if args.save_plots:
        os.makedirs(gv.data_folder, exist_ok=True)
        os.makedirs(gv.fig_folder, exist_ok=True)

    gv.connectivity_simnibs = True
    np.random.seed(gv.random_seed)
    
    gv.stim = False

    print(gv.abeta_stage)
    print(gv.tau_stage)
    print(gv.random_seed)
    print(gv.tms_intensity)
    print(gv.high_tms_freq)

    def actual_weights(ttavg, tavg, stimulus_onset):
        mask_baseline = ttavg < (stimulus_onset-100)
        tavg_baseline  = tavg[mask_baseline]
        amplitude_baseline = np.abs(np.max(tavg_baseline)- np.min(tavg_baseline))
        mask_TMS = ttavg > stimulus_onset
        tavg_TMS  = tavg[mask_TMS]
        amplitude_TMS = np.abs(np.max(tavg_TMS)- np.min(tavg_TMS))
        return amplitude_baseline, amplitude_TMS



    scale_fact = gv.tms_intensity
    start_time = 2500

    weights_vector = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\stimulus_vector_TVB.npy")#np.zeros(86)#
    weights_vector = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\simnibs_simulation_M1_stimulus_vector_TVB.npy")
    weights_vector = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\prova_protocollo_1_stimulus_vector_TVB.npy")
    #print(weights_vector)
    
    train_05Hz = make_train(np.arange(0,86), scale_fact*weights_vector, onset=start_time, T=2000.0, tau=1.0)
    train_10Hz = make_train(np.arange(0,86), scale_fact*weights_vector, onset=start_time, T=100.0, tau=1.0)

    if gv.high_tms_freq:
        stimulus = MultiStimuliRegion(train_10Hz)
    else:
        stimulus = MultiStimuliRegion(train_05Hz)
        
    stimulus = MultiStimuliRegion(train_05Hz)
    stimulus.configure_space()
    time = np.r_[1e3:3e3:15.0]
    stimulus.configure_time(time)
    pattern = stimulus()
    
    sim_time = 4000
    ttavg_stim, tavg_healthy_stim, teeg_stim, eeg_healthy_stim = new_parcellation_simulate(g=gv.coup, velocity=gv.vel, noise=gv.noi, a=gv.aa, b=gv.bb, sim_time=sim_time, stimulus=stimulus, tau_stage=gv.tau_stage, abeta_stage=gv.abeta_stage)#simulate_with_stimulus(stimulus = stimulus, sim_time = sim_time) #Less simulated time needed
    ttavg_stim = ttavg_stim[1000:]
    tavg_stim = tavg_healthy_stim[1000:, :, :, :]
    teeg_stim, eeg_healthy_stim = preprocess(teeg_stim, eeg_healthy_stim)

    amplitude_TMS = []
    amplitude_baseline = []
    for i in range(len(tavg_stim[0, 0, :, 0])):
        amp_b, amp_TMS = actual_weights(ttavg=ttavg_stim, tavg=tavg_stim[:, 0, i, 0], stimulus_onset=start_time)
        amplitude_baseline.append(amp_b)
        amplitude_TMS.append(amp_TMS)
        
    amplitude_baseline = np.array(amplitude_baseline)
    amplitude_TMS = np.array(amplitude_TMS)
    np.save("amplitude_base_stage_11_05_Hz_38_stim.npy", amplitude_baseline)
    np.save("amplitude_TMS_stage_11_05_Hz_38_stim.npy", amplitude_TMS)

if __name__ == "__main__":
    main()
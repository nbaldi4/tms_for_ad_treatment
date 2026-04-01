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
from tms_application_utils import tms_seeds_analysis

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
    parser.add_argument("--spot", type=int, help="0 = PCU, 1 = Fp1, 2 = Fp2, 3 = T3/T7, 4 = T4/T8")
    args = parser.parse_args()

    # Per ogni parametro: se non passato da CLI, chiedi interattivamente
    """
    if args.tau is None:
        args.tau = int(input("Tau stage (0-5): "))

    if args.abeta is None:
        args.abeta = int(input("Beta Amyloid stage (0-5): "))

    if args.rnd is None:
        val = input("Random seed [default: 0]: ")
        args.rnd = int(val) if val.strip() else 0

    if args.tms_intensity is None:
        val = input("Intensity [default: 0]: ")
        args.tms_intensity = float(val) if val.strip() else 0

    if args.spot is None:
        args.spot = int(input("Spot (0 = PCU, 1 = Fp1, 2 = Fp2, 3 = T3/T7, 4 = T4/T8): "))

    if args.high_f and args.low_f:
        raise ValueError('you cannot simulate both high and low intenisty')

    if args.low_f is True:
        gv.high_tms_freq = False
    

    # high_f è un bool (store_true), quindi il default è già False
    # ma se vuoi chiederlo esplicitamente:
    if (not args.high_f) & (not args.low_f) :
        risposta = input("High intensity? (y/n): ").strip().lower()
        args.high_f = risposta == 'y'

    if (args.spot < 0) | (args.spot > 4):
        raise ValueError('Spot codes range from 0 to 4')
    """

    gv.show_plots = args.show_plots
    gv.save_plots = args.save_plots
    if args.rnd:
        gv.random_seed = args.rnd
    if args.tau:
        gv.tau_stage = args.tau
    if args.abeta:
        gv.abeta_stage = args.abeta
    if args.tms_intensity:
        gv.tms_intensity = args.tms_intensity
    gv.high_tms_freq = args.high_f
    if args.spot:
        gv.stimulation_spot = args.spot

    if gv.high_tms_freq:
        freq = 10
    else:
        freq = 0.5

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

    #print(gv.abeta_stage)
    #print(gv.tau_stage)
    #print(gv.random_seed)
    #print(gv.tms_intensity)
    #print(gv.high_tms_freq)
    #print(gv.stimulation_spot)

    def actual_weights(ttavg, tavg, stimulus_onset):
        mask_baseline = ttavg < (stimulus_onset-100)
        tavg_baseline  = tavg[mask_baseline]
        amplitude_baseline = np.abs(np.max(tavg_baseline)- np.min(tavg_baseline))
        mask_TMS = ttavg > stimulus_onset
        tavg_TMS  = tavg[mask_TMS]
        amplitude_TMS = np.abs(np.max(tavg_TMS)- np.min(tavg_TMS))
        return amplitude_baseline, amplitude_TMS

    stimulus_onset = 2500
    weights_vector = np.ones(86)

    if gv.stimulation_spot == 0:
        weights_vector = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\PCU_1_mm_stimulus_vector_TVB.npy")

    if gv.stimulation_spot == 1:
        weights_vector = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\Fp1_1_mm_stimulus_vector_TVB.npy")

    if gv.stimulation_spot == 2:
        weights_vector = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\Fp2_1_mm_stimulus_vector_TVB.npy")

    if gv.stimulation_spot == 3:
        weights_vector = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\T7_T3_1_mm_stimulus_vector_TVB.npy")
        
    if gv.stimulation_spot == 4:
        weights_vector = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\T8_T4_1_mm_stimulus_vector_TVB.npy")

    if gv.high_tms_freq:
        period = 100.0 # time is in ms so f=1/(100/1000)=10 Hz
    else:
        period = 2000.0 # time is in ms so f=1/(2000/1000)=1/2=0.5 Hz
       
    print('period:', period)
    """
    train = make_train(np.arange(0,86), gv.tms_intensity*weights_vector, onset=stimulus_onset, T=period, tau=1.0)
    stimulus = MultiStimuliRegion(train)
    stimulus.configure_space()
    time = np.r_[stimulus_onset - 500:stimulus_onset + 500 + period:0.5]
    stimulus.configure_time(time)
    pattern = stimulus()
    """
    sim_time = stimulus_onset + period + 500
    seeds_number= 2
    tested_values = 2
    threshold = 1.5
    
    seeds_analysis = False
    if seeds_analysis == True:
        results_dir = os.path.join('..', 'Tms_results_3')
        os.makedirs(results_dir, exist_ok=True)
        file_path = os.path.join(results_dir, f"tms_seeds_a{args.abeta}_t{args.tau}_f{freq}_s{gv.stimulation_spot}")
        df = tms_seeds_analysis(file_path, abeta=args.abeta, tau=args.tau, period=period, weights_vector=weights_vector, stimulus_onset=stimulus_onset, tested_values=tested_values, n_seeds=seeds_number)



if __name__ == "__main__":
    main()
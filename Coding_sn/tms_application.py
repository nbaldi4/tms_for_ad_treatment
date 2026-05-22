import logging
import logging.handlers

# Patch per silenziare i PermissionError durante la rotazione dei log
original_handle_error = logging.handlers.TimedRotatingFileHandler.handleError

def silent_handle_error(self, record):
    import sys
    t, v, _ = sys.exc_info()
    if t is PermissionError:
        return  # ignora silenziosamente
    original_handle_error(self, record)

logging.handlers.TimedRotatingFileHandler.handleError = silent_handle_error

import os
import glob

import numpy as np
import pandas as pd
import argparse

from tvb.simulator.lab import *
from matplotlib import pyplot as plt

from classes import MultiStimuliRegion
import global_variable_creation as gv
from tms_application_utils import tms_seeds_analysis, best_sf, infer_stage, stages_boxplots, find_closest_stage, stimulation_analysis, stim_boxplot_stats, no_stim_boxplot_stats, find_all_distances, find_closest_stage3

def main():

    #print('You are going to perform a tms stimulation over a choosen patient (identified by its random seed) in a choosen stage. ' \
    #'You can also select the spot of application (1 mm scalp distance is fixed), the intensity and the frequency of stimulation.' \
    #'Intensity can be any semi-positive number, while frequency of stimulation can only be low (default, 0.5 Hz) or high (10 Hz).')

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
    parser.add_argument("--tms", action="store_true", help="Perform TMS stimulation")
    parser.add_argument("--spot", type=int, help="0 = PCU 1 mm, 2 = Fp1 1 mm, 4 = Fp2 1 mm, 6 = T3/T7 1 mm, 8 = T4/T8 1 mm")
    parser.add_argument("--PSD", action="store_true", help="Perform psd")
    parser.add_argument("--FC", action="store_true", help="Perform fc")
    parser.add_argument("--int_an", action="store_true", help="Perform intensitity analysis")
    parser.add_argument("--best_int", action="store_true", help="Find best superth intensitity")
    parser.add_argument("--build_bp", action="store_true", help="Build boxplots for both stim and no stim stages")
    parser.add_argument("--new_stage", action="store_true", help="Evaluate stages after therapethic threatment")
    parser.add_argument("--vis_test", action="store_true", help="Visualize stimulation example")
    parser.add_argument("--test", action="store_true", help="Parameters value example")
    parser.add_argument("--dis_land", action="store_true", help="Plot disease landscape")
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
    """
    if args.low_f is True:
        gv.high_tms_freq = False

    if args.high_f is True:
        gv.high_tms_freq = True
    """
    
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
    gv.tms_param_variation = args.tms
    gv.fc = args.FC
    gv.psd = args.PSD

    if args.int_an:
        seeds_analysis = True
    else:
        seeds_analysis = False

    if args.test:
        test = True
    else:
        test = False

    if args.dis_land:
        disease_landscape = True
    else:
        disease_landscape = False

    if args.best_int:
        find_best_int = True
    else:
        find_best_int = False

    if args.build_bp:
        build_bp = True
    else:
        build_bp = False

    if args.new_stage:
        infer_new_stage = True
    else:
        infer_new_stage = False

    if args.vis_test:
        vis_test = True
    else:
        vis_test = False

    if args.rnd:
        gv.random_seed = args.rnd
    if args.tau:
        gv.tau_stage = args.tau
    if args.abeta:
        gv.abeta_stage = args.abeta
    if args.tms_intensity:
        gv.tms_intensity = args.tms_intensity

    if args.spot:
        gv.stimulation_spot = args.spot

    if gv.high_tms_freq:
        freq = 10
    else:
        freq = 0.5
    
    gv.frequence = freq

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
        weights_vector = np.load(r"..\Pesi_protocolli_TMS\PCU_1_stimulus_vector_TVB.npy")
        
    if gv.stimulation_spot == 1:
        weights_vector = np.load(r"..\Pesi_protocolli_TMS\PCU_10_stimulus_vector_TVB.npy")

    if gv.stimulation_spot == 2:
        weights_vector = np.load(r"..\Pesi_protocolli_TMS\Fp1_1_stimulus_vector_TVB.npy")

    if gv.stimulation_spot == 3:
        weights_vector = np.load(r"..\Pesi_protocolli_TMS\Fp1_10_stimulus_vector_TVB.npy")

    if gv.stimulation_spot == 4:
        weights_vector = np.load(r"..\Pesi_protocolli_TMS\Fp2_1_stimulus_vector_TVB.npy")

    if gv.stimulation_spot == 5:
        weights_vector = np.load(r"..\Pesi_protocolli_TMS\Fp2_10_stimulus_vector_TVB.npy")

    if gv.stimulation_spot == 6:
        weights_vector = np.load(r"..\Pesi_protocolli_TMS\T7T3_1_stimulus_vector_TVB.npy")

    if gv.stimulation_spot == 7:
        weights_vector = np.load(r"..\Pesi_protocolli_TMS\T7T3_10_stimulus_vector_TVB.npy")
        
    if gv.stimulation_spot == 8:
        weights_vector = np.load(r"..\Pesi_protocolli_TMS\T8T4_1_stimulus_vector_TVB.npy")

    if gv.stimulation_spot == 9:
        weights_vector = np.load(r"..\Pesi_protocolli_TMS\T8T4_10_stimulus_vector_TVB.npy")

    #print(weights_vector)

    if gv.high_tms_freq:
        period = 100.0 # time is in ms so f=1/(100/1000)=10 Hz
    else:
        period = 2000.0 # time is in ms so f=1/(2000/1000)=1/2=0.5 Hz

    #print('period', period)
       
    """
    train = make_train(np.arange(0,86), gv.tms_intensity*weights_vector, onset=stimulus_onset, T=period, tau=1.0)
    stimulus = MultiStimuliRegion(train)
    stimulus.configure_space()
    time = np.r_[stimulus_onset - 500:stimulus_onset + 500 + period:0.5]
    stimulus.configure_time(time)
    pattern = stimulus()
    """

    seeds_number= 4
    tested_values = 28
    threshold = 1.5

    
    from utils import preprocess, psd, functional_connectivity_band, make_train, new_parcellation_simulate
    if gv.tms_param_variation == True:
        ttavg, tavg, teeg, eeg = new_parcellation_simulate(g=gv.coup, velocity=gv.vel, noise=gv.noi, a=gv.aa, b=gv.bb, factor=gv.factor, abeta_stage=gv.abeta_stage, tau_stage=gv.tau_stage)
        ttavg, tavg = preprocess(ttavg, tavg)
        teeg, eeg = preprocess(teeg, eeg)
        if gv.fc == True:
            #functional_connectivity(tavg, label='tavg')
            
            #functional_connectivity_lorenzo(tavg)
            functional_connectivity_band(tavg, label='TAVG', lowcut=0.5, highcut=45)
            functional_connectivity_band(tavg, label='TAVG delta', lowcut=0.5, highcut=4)
            functional_connectivity_band(tavg, label='TAVG theta', lowcut=4, highcut=8)
            functional_connectivity_band(tavg, label='TAVG alpha', lowcut=8, highcut=13)
            functional_connectivity_band(tavg, label='TAVG beta', lowcut=13, highcut=30)
            #functional_connectivity_band(eeg, label='EEG', lowcut=0.5, highcut=45)
            #functional_connectivity_band(eeg, label='EEG delta', lowcut=0.5, highcut=4)
            #functional_connectivity_band(eeg, label='EEG theta', lowcut=4, highcut=8)
            #functional_connectivity_band(eeg, label='EEG alpha', lowcut=8, highcut=13)
            #functional_connectivity_band(eeg, label='EEG beta', lowcut=13, highcut=30)
            
            #functional_connectivity_img_coh(eeg, 'EEG beta', lowcut=13, highcut=30)
            #functional_connectivity_img_coh(eeg, 'EEG alpha', lowcut=8, highcut=13)
            #functional_connectivity_img_coh(eeg, 'EEG theta', lowcut=4, highcut=8)
            #functional_connectivity_img_coh(eeg, 'EEG delta', lowcut=0.5, highcut=4)
            #functional_connectivity_img_coh(tavg, 'TAVG beta', lowcut=13, highcut=30)
            #functional_connectivity_img_coh(tavg, 'TAVG alpha', lowcut=8, highcut=13)
            #functional_connectivity_img_coh(tavg, 'TAVG theta', lowcut=4, highcut=8)
            #functional_connectivity_img_coh(tavg, 'TAVG delta', lowcut=0.5, highcut=4)
            #functional_connectivity_img_coh(eeg, 'EEG', lowcut=0.5, highcut=45)
            #functional_connectivity_img_coh(tavg, 'TAVG', lowcut=0.5, highcut=45)
        
        if gv.psd == True:
            psd(eeg, 'EEG')
            psd(tavg, 'TAVG')

    if seeds_analysis == True:
        results_dir = os.path.join('..', 'Tms_results_end')
        os.makedirs(results_dir, exist_ok=True)
        file_path = os.path.join(results_dir, f"tms_seeds_a{args.abeta}_t{args.tau}_f{freq}_s{gv.stimulation_spot}")
        df = tms_seeds_analysis(file_path, abeta=args.abeta, tau=args.tau, period=period, weights_vector=weights_vector, stimulus_onset=stimulus_onset, tested_values=tested_values, n_seeds=seeds_number, freq=gv.high_tms_freq)

    if find_best_int == True:
        base_dir = r"..\Tms_results_end_18"
        os.makedirs(base_dir, exist_ok=True)
        files = glob.glob(os.path.join(base_dir, "tms_seeds_a*_t*_f*_s*"))
        print(len(files))
        sf_values = []
        for file in files:
            #print(best)
            #print(bool_arr[35:55]) 
            filename = os.path.basename(file)
            params = filename.replace("tms_seeds_", "")
            output_name = f"super_th_reg_{params}"
            output_path = os.path.join(base_dir, output_name)
            df= pd.read_pickle(file)#r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Tms_app_results_2\tms_seeds_a3_t3_f0.5_s0")
            df_2, best_sf_value, best_n_channels = best_sf(df, amplitude_threshold_up=18)
            df_2.to_csv(output_path)
            df_2.to_pickle(output_path) 
            sf_values.append((params, best_sf_value, best_n_channels))
        #sf_df = pd.DataFrame(sf_values, columns=['params', 'best_sf', 'best_n_channels'])
        #sf_df.to_csv(os.path.join(base_dir, 'best_sf_values.csv'), index=False)
        #sf_df.to_pickle(os.path.join(base_dir, 'best_sf_values.pkl'))

    cure_folder = 'Compare_cure_n02_th18_new_var5'
    results_folder = 'n02_th10_2'
    if build_bp == True:
        new_bp_stages = False
        save_dir = os.path.join(r"..\Results_data", cure_folder)
        os.makedirs(save_dir, exist_ok=True)  # crea la cartella se non esiste, senza errori se esiste già
        if new_bp_stages == True:
            df_boxplots = stages_boxplots()

            df_boxplots = df_boxplots.rename(columns={
                'FC_abs_mean_broad': 'Fbr',
                'FC_abs_mean_delta': 'Fd',
                'FC_abs_mean_theta': 'Ft',
                'FC_abs_mean_alpha': 'Fa',
                'FC_abs_mean_beta' : 'Fbe',
                'PSD_delta_over_br': 'Pdobr',
                'PSD_theta_over_br': 'Ptobr',
                'PSD_alpha_over_br': 'Paobr',
                'PSD_beta_over_br' : 'Pbobr',
            })
            df_boxplots.to_csv(os.path.join(save_dir, 'no_stim_stages.csv'), index=False)
            df_boxplots.to_pickle(os.path.join(save_dir, 'no_stim_stages.pkl'))

            df = no_stim_boxplot_stats(df_boxplots)
            df.to_csv(os.path.join(save_dir, 'no_stim_box.csv'), index=False)
            df.to_pickle(os.path.join(save_dir, 'no_stim_box.pkl'))

        df_stages = infer_stage(
            root = os.path.join(r"..\Results_data",results_folder), 
            dir_pattern = r"Ab(\d+)t(\d+)s(\d+)f--(\w+)_fi(\d+(?:\.\d+)?)[/\\]tau(\d+)_abeta(\d+)_rs(\d+)"
            )
        df_stages = df_stages.rename(columns={
            'FC_abs_mean_broad': 'Fbr',
            'FC_abs_mean_delta': 'Fd',
            'FC_abs_mean_theta': 'Ft',
            'FC_abs_mean_alpha': 'Fa',
            'FC_abs_mean_beta' : 'Fbe',
            'PSD_delta_over_br': 'Pdobr',
            'PSD_theta_over_br': 'Ptobr',
            'PSD_alpha_over_br': 'Paobr',
            'PSD_beta_over_br' : 'Pbobr',
        })
        df_stages.to_csv(os.path.join(save_dir, 'stim_stages.csv'), index=False)
        df_stages.to_pickle(os.path.join(save_dir, 'stim_stages.pkl'))

        df = stim_boxplot_stats(df_stages)
        df.to_csv(os.path.join(save_dir, 'stim_box.csv'), index=False)
        df.to_pickle(os.path.join(save_dir, 'stim_box.pkl'))

        save_dir = os.path.join(r"..\Results_data", cure_folder)
        stim_box = pd.read_pickle(os.path.join(save_dir, 'stim_box.pkl'))
        no_stim_box = pd.read_pickle(r"..\Results_data\No_stim_box_plots\no_stim_box.pkl")
        df_closest = find_closest_stage(stim_box, no_stim_box, metrics=['Fa', 'Fbe', 'Pdobr', 'Ptobr', 'Paobr', 'Pbobr'])
        df_closest.to_csv(os.path.join(save_dir, 'closest_stage_Fa_Fbe_Pdobr_Ptobr_Paobr_Pbobr_norm.csv'), index=False)
        df_closest.to_pickle(os.path.join(save_dir, 'closest_stage_Fa_Fbe_Pdobr_Ptobr_Paobr_Pbobr_norm.pkl'))
        df_closest_all_pairs = find_all_distances(stim_box, no_stim_box, metrics=['Fa', 'Fbe','Pdobr', 'Ptobr', 'Paobr', 'Pbobr'])
        df_closest_all_pairs.to_csv(os.path.join(save_dir, 'closest_stage_all_pairs_Fa_Fbe_Pdobr_Ptobr_Paobr_Pbobr_norm.csv'), index=False)
        df_closest_all_pairs.to_pickle(os.path.join(save_dir, 'closest_stage_all_pairs_Fa_Fbe_Pdobr_Ptobr_Paobr_Pbobr_norm.pkl'))

    if disease_landscape == True:
        no_stim_box = pd.read_pickle(r"..\Results_data\No_stim_box_plots\no_stim_box.pkl")
        metrics = ['Pdobr', 'Ptobr', 'Paobr', 'Pbobr']#['Fa', 'Fbe', 'Pdobr', 'Ptobr', 'Paobr', 'Pbobr']#, 'Fd', 'Ft', 'Fa', 'Fbe', 'Pdobr', 'Ptobr', 'Paobr', 'Pbobr']
        suffix = '_med'
        distances = np.zeros((6,6))
        save_dir = os.path.join(r"..\Results_data", cure_folder)
        stim_box = pd.read_pickle(os.path.join(save_dir, 'stim_box.pkl'))
        #find_closest_stage3(stim_box, no_stim_box=no_stim_box, metrics=metrics)

        normalization = {}
        for m in metrics:
            col = f'{m}{suffix}'
            rng = np.mean(no_stim_box[col])
            normalization[m] = rng if rng != 0 else 1.0
        
        print(normalization)

        for _, ns_row in no_stim_box.iterrows():
                
            current_dist = 0
            for m in metrics:
                col = f'{m}{suffix}'
                diff = np.abs(ns_row[col]) / normalization[m]
                current_dist += diff
            
            abeta = int(ns_row['b'])
            tau = int(ns_row['t'])
            distances[abeta, tau] = current_dist
        
        print(distances)
        fig, ax = plt.subplots(figsize=(12, 8))

        im = ax.imshow(distances, interpolation='nearest', cmap='binary')
        ax.set_xlabel('tau')
        ax.set_ylabel('abeta')

        fig.colorbar(im, ax=ax)

        plt.close()
        

        distances_stim = []

        for _, s_row in stim_box.iterrows():
                
            current_dist = 0
            for m in metrics:
                col = f'{m}{suffix}'
                diff = np.abs(s_row[col]) / normalization[m]
                current_dist += diff
            
            abeta = int(s_row['b'])
            tau = int(s_row['t'])
            spot = int(s_row['sp'])
            freq = s_row['f']
            if freq == 'high':
                freqindex = 0
            else:
                freqindex = 1
            inte = int(s_row['inte'])
            if inte == 1:
                inteindex = 0
            if inte == 10:
                inteindex = 1
            if inte == 100:
                inteindex = 2
            if inte == 500:
                inteindex = 3
            distances_stim.append((abeta, tau, spot, freq, inte, current_dist))
        df_stim_dist = pd.DataFrame(distances_stim, columns=['abeta', 'tau', 'spot', 'freq', 'inte', 'current_dist'])

        #print(df_stim_dist.head())
        df_stim_dist.to_pickle(os.path.join(r"..", 'test.pkl'))

        #mask = distances_stim > 0
        #print(distances_stim)

    if vis_test == True:

        sim_time = stimulus_onset + 2000 + 500

        intens = 500

        weights_vector = np.load(r"..\Pesi_protocolli_TMS\T8T4_1_stimulus_vector_TVB.npy")
        train = make_train(
            np.arange(0, 86),
            intens * weights_vector,
            onset=stimulus_onset,
            T=period,
            tau=1.0
        )

        stimulus = MultiStimuliRegion(train)
        stimulus.configure_space()
        time = np.r_[stimulus_onset - 500:stimulus_onset + 500 + 2000:0.5]
        stimulus.configure_time(time)
        pattern = stimulus()
        ttavg_stim, tavg_stim, teeg_stim, eeg_stim = new_parcellation_simulate(
            g=gv.coup,
            velocity=gv.vel,
            noise=gv.noi,
            a=gv.aa,
            b=gv.bb,
            sim_time=sim_time,
            stimulus=stimulus,
            tau_stage=5,
            abeta_stage=5
        )

        ttavg_stim = ttavg_stim[1000:]
        tavg_stim = tavg_stim[1000:, :, :, :]
        teeg_stim = teeg_stim[1000:]
        eeg_stim = eeg_stim[1000:, :, :, :]

        signal = tavg_stim
        time_array = ttavg_stim

        channel = 33

        baseline, pre_amp, post_amp = stimulation_analysis(
            times=time_array,
            timeseries=signal[:, 0, channel, 0],
            stimulus_onset=stimulus_onset
        )

        print(baseline)
        print(pre_amp)
        print(post_amp)
        plt.plot(time_array, signal[:,0,channel,0])
        plt.plot(time_array, baseline*np.ones(len(time_array)))
        plt.plot(time_array, (baseline-pre_amp)*np.ones(len(time_array)))
        plt.plot(time_array, (baseline+pre_amp)*np.ones(len(time_array)))
        plt.plot(time_array, (baseline-post_amp)*np.ones(len(time_array)))
        plt.plot(time_array, (baseline+post_amp)*np.ones(len(time_array)))
        plt.show()

    if test == True:

        sim_time = 5000

        intens = 10**5

        train = make_train(
            np.arange(0, 86),
            intens * weights_vector,
            onset=stimulus_onset,
            T=period,
            tau=1.0
        )

        stimulus = MultiStimuliRegion(train)
        stimulus.configure_space()
        time = np.r_[stimulus_onset - 500:stimulus_onset + 500 + 2000:0.5]
        stimulus.configure_time(time)
        pattern = stimulus()
        ttavg_stim, tavg_stim, teeg_stim, eeg_stim = new_parcellation_simulate(
            sim_time=sim_time,
            stimulus=stimulus
        )


if __name__ == "__main__":
    main()
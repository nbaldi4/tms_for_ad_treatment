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

from classes import JRPSP
from global_variable_creation import get_eeg_monitor, get_conn
import global_variable_creation as gv
from additional_utils import macro_fc

file_path = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Stages_raw_data\tau0_abeta0_rnd0\eeg_and_teeg_raw_tau0_abeta0_rnd0.npz")
eeg = file_path['eeg']
teeg = file_path['teeg']


def preprocess(times, time_serie, start_time=1000):
    """
    Normalize and re-reference EEG data.

    Args:
        times (np.ndarray): times array.
        time_serie (np.ndarray): time_serie signal array of shape [time, 1, channels, 1].
        start_time (int): Index at which to start preprocessing.

    Returns:
        np.ndarray: Preprocessed times (cut from start time to end).
        np.ndarray: Preprocessed data (normalized, zero-mean).
    """
    times = times[start_time:]
    time_serie = time_serie[start_time:, :, :, :]
    trange = np.max(time_serie, axis=0) - np.min(time_serie, axis=0)
    trange[trange == 0] = 1
    time_serie = (time_serie - np.min(time_serie, axis=0)) / trange
    return times, time_serie

def bandpass_filter(data, lowcut, highcut, fs=1000.0, order=4):
    """
    Apply a zero-phase Butterworth bandpass filter.

    Args:
        data (np.ndarray): 1D signal array.
        lowcut (float): Lower cutoff frequency (Hz).
        highcut (float): Upper cutoff frequency (Hz).
        fs (float): Sampling frequency (Hz). Default = 1000.
        order (int): Filter order.

    Returns:
        np.ndarray: Filtered signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def psd(time_series_original, label, lowcut=0.5, highcut=45, n_seg=2048, window='hamming', return_psd=False, real_patient = False, patient = 0):
    """
    Compute and visualize the Power Spectral Density (PSD) of regional brain signals.

    Args:
        time_series (np.ndarray): Array [time, 1, regions, 1].
        label (str): Dataset label for plotting.
        lowcut (float): Lower cutoff frequency for filtering (Hz).
        highcut (float): Upper cutoff frequency (Hz).
        n_seg (int): Segment length for Welch method.
        window (str): Window function name (e.g., 'hamming').

    Notes:
        - The PSD is computed via Welch’s method for each region.
        - The mean and standard deviation across regions are plotted.
    """
    time_series = time_series_original.copy()
    n_timepoints = time_series.shape[0]
    n_regions = time_series.shape[2]
    filtered_signals = np.zeros((n_timepoints, n_regions))

    # Apply bandpass filter
    for i in range(n_regions):
        region_signal = time_series[:, 0, i, 0]
        filtered_signals[:, i] = bandpass_filter(region_signal, lowcut, highcut, order=4)

    # Compute PSD for each region
    tau_stage_1 = [
        24, 62   # PHC
    ]

    tau_stage_2 = [
        9, 47,       # Temporal allocortex / hippocampal limbic III -> HC
        31, 69,      # TCI
        30, 68,      # TCC
        34, 72       # TCV
    ]
    tau_stage_3 = [
        18, 56,   # PFCDL
        17, 55,   # PFCCL
        19, 57,   # PFCDM
        12, 50,   # M1
        28, 66,   # S1
        29, 67,   # S2
        35, 73,   # V1 
        36, 74    # V2
    ]
    psd_all = []
    fs_welch = 1 / (gv.temp_avg_period / 1000)
    for i in range(n_regions):
        freqs, psd_vals = signal.welch(
            filtered_signals[:, i],
            fs=fs_welch,
            nperseg=min(n_seg, len(filtered_signals[:, i])),
            noverlap=n_seg // 2,
            window=window
        )
        psd_all.append(psd_vals)
    #print(f"Range frequenze: da {freqs[0]} Hz a {freqs[-1]} Hz")
    tau_stages = tau_stage_3+tau_stage_2+tau_stage_1

    psd_all = np.array(psd_all)
    psd_mean = np.mean(psd_all, axis=0)
    psd_std = np.std(psd_all, axis=0)

    # psd_all array dimensions are [n_regions, n_freqs]

    if len(filtered_signals[0, :]) > 75:
        psd_affected = []
        psd_non_affected = []
        tau_stage_all = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 20, 21, 22, 23, 25, 26, 27, 32, 33, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 51, 52, 53, 54, 58, 59, 60, 61, 63, 64, 65, 70, 71, 75
        ]
        for i in tau_stage_all:
            freqs_non_affected, psd_vals = signal.welch(
                filtered_signals[:, i],
                fs=fs_welch,
                nperseg=min(n_seg, len(filtered_signals[:, i])),
                noverlap=n_seg // 2,
                window=window
            )
            psd_non_affected.append(psd_vals)

        psd_non_affected = np.array(psd_non_affected)
        psd_non_affected_mean = np.mean(psd_non_affected, axis=0)
        psd_non_affected_std = np.std(psd_non_affected, axis=0)

        non_affected_normalization_factor = np.max(psd_non_affected_mean)
        psd_non_affected_mean /= non_affected_normalization_factor
        psd_non_affected_std /= non_affected_normalization_factor
        
        plt.figure(figsize=(10, 6))
        plt.plot(freqs_non_affected, psd_non_affected_mean, label=label+' non affected', linewidth=3)
        plt.fill_between(freqs_non_affected, psd_non_affected_mean + psd_non_affected_std, psd_non_affected_mean - psd_non_affected_std, alpha=0.35)
        plt.title(label + ' non affected regions mean PSD', fontsize=20)
        plt.ylim(0, 1.01)
        plt.xlim(0, 45)
        plt.xlabel('Frequency [Hz]', fontsize=16)
        plt.ylabel('Normalized PSD', fontsize=16)
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.show()

        tau_stage_all = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 20, 21, 22, 23, 25, 26, 27, 32, 33, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 51, 52, 53, 54, 58, 59, 60, 61, 63, 64, 65, 70, 71, 75
        ]
        for i in tau_stages:
            freqs_affected, psd_vals = signal.welch(
                filtered_signals[:, i],
                fs=fs_welch,
                nperseg=min(n_seg, len(filtered_signals[:, i])),
                noverlap=n_seg // 2,
                window=window
            )
            psd_affected.append(psd_vals)

        psd_affected = np.array(psd_affected)
        psd_affected_mean = np.mean(psd_affected, axis=0)
        psd_affected_std = np.std(psd_affected, axis=0)

        affected_normalization_factor = np.max(psd_affected_mean)
        psd_affected_mean /= affected_normalization_factor
        psd_affected_std /= affected_normalization_factor

        plt.figure(figsize=(10, 6))
        plt.plot(freqs_affected, psd_affected_mean, label=label+' affected', linewidth=3)
        plt.fill_between(freqs_affected, psd_affected_mean + psd_affected_std, psd_affected_mean - psd_affected_std, alpha=0.35)
        plt.title(label + ' affected regions mean PSD', fontsize=20)
        plt.ylim(0, 1.01)
        plt.xlim(0, 45)
        plt.xlabel('Frequency [Hz]', fontsize=16)
        plt.ylabel('Normalized PSD', fontsize=16)
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.show()

    # Normalize
    normalization_factor = np.max(psd_mean)
    psd_mean_normalized = psd_mean.copy()
    psd_mean_normalized /= normalization_factor
    #psd_std /= normalization_factor

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, psd_mean, label=label, linewidth=3)
    plt.fill_between(freqs, psd_mean + psd_std, psd_mean - psd_std, alpha=0.35)
    plt.title(label + ' mean PSD', fontsize=20)
    #plt.ylim(0, 1.01)
    plt.xlim(0, 45)
    plt.xlabel('Frequency [Hz]', fontsize=16)
    plt.ylabel('PSD che dovrebbe essere giusta', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.show()

    if len(filtered_signals[0, :]) < 70:    #P4(6), P3(7), O2(8), O1(9), T8(12), T7(13), Pz(16) regions probably more affected by AD
        psd_interested = []
        interested_nodes = [
            6, 7, 8, 9, 12, 13, 16
        ]
        for i in interested_nodes:
            freqs_AD, psd_AD = signal.welch(
                filtered_signals[:, i],
                fs=fs_welch,
                nperseg=min(n_seg, len(filtered_signals[:, i])),
                noverlap=n_seg // 2,
                window=window
            )
            psd_interested.append(psd_AD)

        psd_interested = np.array(psd_interested)
        psd_interested_mean = np.mean(psd_interested, axis=0)
        psd_interested_std = np.std(psd_interested, axis=0)

        #if real_patient == False:
        #    fig_save_dir = os.path.join(gv.fig_folder, f"tau{gv.tau_stage}_abeta{gv.abeta_stage}_a{gv.aa:.4f}_b{gv.bb:.4f}_g{gv.coup:.4f}_v{gv.vel:.2f}_n{gv.noi:.4f}_simtime{gv.sim_time}_rs{gv.random_seed}") #gv.fig_folder
        #    os.makedirs(fig_save_dir, exist_ok=True)
        #    data_save_dir = os.path.join(gv.data_folder, f"tau{gv.tau_stage}_abeta{gv.abeta_stage}_a{gv.aa:.4f}_b{gv.bb:.4f}_g{gv.coup:.4f}_v{gv.vel:.2f}_n{gv.noi:.4f}_simtime{gv.sim_time}_rs{gv.random_seed}")  #gv.data_folder
        #    os.makedirs(data_save_dir, exist_ok=True)
            # Normalize
        normalization_factor = np.max(psd_interested_mean)
        #psd_mean /= normalization_factor
        #psd_std /= normalization_factor

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(freqs_AD, psd_interested_mean, label=label, linewidth=3)
        plt.fill_between(freqs_AD, psd_interested_mean + psd_interested_std, psd_interested_mean - psd_interested_std, alpha=0.35)
        plt.title(label + ' AD regions mean PSD', fontsize=20)
        #plt.ylim(0, 1.01)
        plt.xlim(0, 45)
        plt.xlabel('Frequency [Hz]', fontsize=16)
        plt.ylabel('PSD', fontsize=16)
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.show()

    if return_psd:
        return freqs, psd_mean_normalized

def ev_img_coh(time_series_original, lowcut, highcut):
    conn = get_conn()
    time_series = time_series_original.copy()
    # TVB 4D format: [Time, Var, Nodes, Modes]
    # Squeeziamo per avere [Time, Nodes] assumendo Var=0 e Mode=0
    data = time_series[:, 0, :, 0]
    n_nodes = data.shape[1]
    fs = 1000 / gv.temp_avg_period  # Frequenza di campionamento in Hz
    # Inizializziamo la matrice iCOH
    icoh_matrix = np.zeros((n_nodes, n_nodes))

    # Calcoliamo la coerenza per ogni coppia di nodi
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes): # Solo triangolo superiore per velocità
            # Calcolo della coerenza complessa
            # Nota: scipy.signal.coherence restituisce la magnitudo quadra. 
            # Per la coerenza immaginaria serve la Cross-Spectral Density (CSD).
            
            f, Pxy = csd(data[:, i], data[:, j], fs=fs, nperseg=int(fs*2)) # finestre di 2s
            f, Pxx = welch(data[:, i], fs=fs, nperseg=int(fs*2))
            f, Pyy = welch(data[:, j], fs=fs, nperseg=int(fs*2))
            
            # Coerenza Complessa = Pxy / sqrt(Pxx * Pyy)
            complex_coherence = Pxy / np.sqrt(Pxx * Pyy)
            
            # Estraiamo la parte immaginaria
            # Spesso si fa la media sulla banda di interesse (es. Alpha 8-13Hz)
            band_mask = (f > lowcut) & (f <= highcut)
            icoh_val = np.mean(np.imag(complex_coherence[band_mask]))
            
            icoh_matrix[i, j] = icoh_val
            icoh_matrix[j, i] = icoh_val # Simmetria

    # Per mantenere compatibilità con il tuo oggetto TimeSeriesRegion e graph.CorrelationCoefficients
    # creiamo un array 4D finto come si aspetta il resto del tuo codice
    result = icoh_matrix[:, :, np.newaxis, np.newaxis]
    
    # Ricostruiamo l'oggetto TVB (anche se i nomi dicono 'corr', conterrà iCOH)
    tsr = TimeSeriesRegion(connectivity=conn, data=time_series, sample_period=gv.temp_avg_period)
    tsr.configure()
    corr_coeff = graph.CorrelationCoefficients(source=tsr, array_data=result)
    
    return corr_coeff

def functional_connectivity_img_coh(time_series_original, label, lowcut, highcut, patient = 0, real_patient = False):

    if real_patient == False:
        fig_save_dir = os.path.join(gv.fig_folder, f"tau{gv.tau_stage}_abeta{gv.abeta_stage}_a{gv.aa:.4f}_b{gv.bb:.4f}_g{gv.coup:.4f}_v{gv.vel:.2f}_n{gv.noi:.4f}_simtime{gv.sim_time}_rs{gv.random_seed}") #gv.fig_folder
        os.makedirs(fig_save_dir, exist_ok=True)
    else:
        fig_save_dir = os.path.join(gv.fig_folder, f"patient_{patient}")
        os.makedirs(fig_save_dir, exist_ok=True)

    conn = get_conn()
    time_series = time_series_original.copy()
    #time_series = time_series[np.isnan(time_series)]
    tsr_corr = ev_img_coh(time_series, lowcut, highcut)

    corr= tsr_corr.array_data[..., 0, 0]
    abs_corr_mean = np.mean(np.abs(corr))
    corr_mean = np.mean(corr)
    sub_corr = macro_fc(corr)
    abs_sub_corr_mean = np.mean(np.abs(sub_corr))
    corr_sub_mean = np.mean(sub_corr)

    #Define a threshold for connections
    corr_th = (corr) > gv.th

    """
    pmatrix = np.zeros((len(time_series[0,0,:,0]),len(time_series[0,0,:,0])))

    for i in range(len(time_series[0,0,:,0])):
        for j in range(len(time_series[0,0,:,0])):
            
            region_A = time_series[:,0, i, 0]
            region_B = time_series[:,0, j, 0]
            
            pmatrix[i][j] = pearsonr(region_A, region_B)[1]

    pmatrix_th = pmatrix < 0.05
    corr_th_significance = np.multiply(corr_th, pmatrix_th)
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    corr_viz = 2 * (corr - np.min(corr)) / (np.max(corr) - np.min(corr)) - 1
    # Plot corr_tavg
    im1 = ax1.imshow(corr_viz, interpolation='nearest', cmap='bwr')
    ax1.set_yticks(np.arange(len(conn.region_labels)))
    ax1.set_yticklabels(conn.region_labels, fontsize=12)
    ax1.set_xticks(np.arange(len(conn.region_labels)))
    ax1.set_xticklabels(conn.region_labels, rotation='vertical', fontsize=12)
    if real_patient == False:
        ax1.set_title(f"{label} FC mean = {corr_mean:.4f} {lowcut}-{highcut} Hz, ({abs_corr_mean:.4f})", fontsize=15)
    else:
        ax1.set_title(f"Patient {patient} {lowcut}-{highcut} Hz FC mean = {corr_mean:.4f}, ({abs_corr_mean:.4f})", fontsize=15)
    # Plot corr_tavg_th
    im2 = ax2.imshow(corr_th, interpolation='nearest', cmap='bwr')
    ax2.set_yticks(np.arange(len(conn.region_labels)))
    ax2.set_yticklabels(conn.region_labels, fontsize=12)
    ax2.set_xticks(np.arange(len(conn.region_labels)))
    ax2.set_xticklabels(conn.region_labels, rotation='vertical', fontsize=12)
    if real_patient == False:
        ax2.set_title(label+f' FC w/ threshold {lowcut}-{highcut} Hz ', fontsize=15)
    else:
        ax2.set_title(f'FC w/ threshold {lowcut}-{highcut} Hz ', fontsize=15)

    # Colorbar
    cax = fig.add_axes([ax2.get_position().x1 + 0.01, ax2.get_position().y0, 0.02, ax2.get_position().height])
    cbar = plt.colorbar(im2, cax=cax)
    plt.show()    

    sub_corr_viz = 2 * (sub_corr - np.min(sub_corr)) / (np.max(sub_corr) - np.min(sub_corr)) - 1
    # Plot corr
    fig = plt.figure(figsize=(12, 8)) 
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
    im = ax.imshow(sub_corr_viz, cmap='bwr', vmin=-1, vmax=1, aspect='equal') 
    if real_patient == False:
        title_text = f"Reduced size simulated FC\n{label} FC reduced mean = {corr_sub_mean:.4f}, ({abs_sub_corr_mean:.4f})"
    else:
        title_text = f"Patient {patient} reduced size FC reduced mean = {corr_sub_mean:.4f}, ({abs_sub_corr_mean:.4f})"
    ax.set_title(title_text, fontsize=15)
    cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
    plt.colorbar(im, cax=cax)
    plt.show()

eeg, eeg = preprocess(teeg, eeg)
psd(eeg, label='test')
functional_connectivity_img_coh(eeg, label='test', lowcut=0.5, highcut=45)

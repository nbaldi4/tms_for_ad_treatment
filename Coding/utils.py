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
from scipy.signal import butter, filtfilt
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


def skin_and_sensors():
    """
    Visualize the patient's skin surface, EEG sensor positions, and the connectivity matrix.

    This function loads the surface (skin) and EEG sensor configuration from file,
    visualizes them in 3D, and plots the brain connectivity matrix.

    It saves and/or displays the plots depending on the global variables:
        - save_plots
        - show_plots
        - fig_folder
    """
    conn = get_conn()

    # Load and configure patient skin model
    skin = surfaces.SkinAir.from_file()
    skin.configure()

    # Load and configure EEG sensor positions
    sens_eeg = sensors.SensorsEEG.from_file()
    sens_eeg.configure()

    # === 3D Visualization of skin and sensors ===
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection='3d')

    # Plot cortical regions as black dots
    x, y, z = conn.centres.T[:, :]
    ax.plot(x, y, z, 'ko')

    # Plot EEG sensors as red X's
    x, y, z = sens_eeg.sensors_to_surface(skin).T
    ax.plot(x, y, z, 'rx')

    # Plot the surface mesh
    x, y, z = skin.vertices.T
    ax.plot_trisurf(x, y, z, triangles=skin.triangles, alpha=0.1, edgecolor='k')

    # Save or display
    if gv.save_plots:
        fig_name = "Skin_and_sensors.png"
        os.makedirs(gv.fig_folder, exist_ok=True)
        plt.savefig(os.path.join(gv.fig_folder, fig_name), dpi=200)
    if gv.show_plots:
        plt.show()

    # === Connectivity Matrix ===
    plt.figure()
    plt.imshow(conn.weights, cmap='binary')

    if gv.save_plots:
        fig_name = "Connectivity.png"
        plt.savefig(os.path.join(gv.fig_folder, fig_name), dpi=200)
    if gv.show_plots:
        plt.show()

def brain_activity(eeg, tavg, start_time=-1000):
    """
    Visualize EEG and regional brain activity signals.

    Args:
        eeg (np.ndarray): EEG signals, shape [time, channels].
        tavg (np.ndarray): Region-averaged signals, shape [time, regions].
        start_time (int): Time index from which to start visualization.

    Normalizes and re-references the data before plotting.
    """
    conn = get_conn()
    mon_eeg = get_eeg_monitor()

    # === EEG signals ===
    eeg = eeg[start_time:, :]
    eeg /= (np.max(eeg, 0) - np.min(eeg, 0))
    eeg -= np.mean(eeg, 0)
    times_eeg = np.arange(eeg.shape[0])

    # === Regional signals ===
    tavg = tavg[start_time:]
    tavg /= (np.max(tavg, 0) - np.min(tavg, 0))
    tavg -= np.mean(tavg, 0)
    times_tavg = np.arange(tavg.shape[0])

    # EEG Plot
    plt.figure(figsize=(10, 12))
    plt.plot(times_eeg, eeg + np.r_[:len(mon_eeg.sensors.labels)], color='b')
    plt.yticks(np.r_[:len(mon_eeg.sensors.labels)], mon_eeg.sensors.labels, fontsize=12)
    plt.title("Brain activity (EEG)", fontsize=18)
    plt.xlabel('Time (ms)', fontsize=16)

    if gv.save_plots:
        plt.savefig(os.path.join(gv.fig_folder, "Brain_activity_EEG.png"), dpi=200)
    if gv.show_plots:
        plt.show()

    # Regional activity Plot
    plt.figure(figsize=(10, 12))
    plt.plot(times_tavg, tavg + np.r_[:len(conn.region_labels)], color='red')
    plt.yticks(np.r_[:len(conn.region_labels)], conn.region_labels, fontsize=12)
    plt.title("Brain activity (region space)", fontsize=18)
    plt.xlabel('Time (ms)', fontsize=16)

    if gv.save_plots:
        plt.savefig(os.path.join(gv.fig_folder, "Brain_activity_region_space.png"), dpi=200)
    if gv.show_plots:
        plt.show()

def eeg_plot(datas, times, display_time=3000, chlist=[4, 13, 20, 23, 38, 41, 45, 50, 62]):
    """
    Visualize EEG signals for multiple conditions or datasets.

    Args:
        datas (dict): Dictionary where each key is a condition, containing:
            - 'eeg': EEG data array
            - 'label': Name of the condition
            - 'color': Line color
        times (np.ndarray): Time points (in ms)
        display_time (int): Number of milliseconds to display
        chlist (list[int]): List of EEG channels to plot
    """
    mon_eeg = get_eeg_monitor()

    for key, value in datas.items():
        time_series = value['eeg']
        label = value['label']
        color = value['color']

        plt.figure(figsize=(10, 12))
        plt.plot(times[:display_time],
                 time_series[:display_time, 0, chlist, 0] + np.r_[:len(chlist)],
                 color=color)
        plt.yticks(np.r_[:len(chlist)], mon_eeg.sensors.labels[chlist], fontsize=12)
        plt.title("EEG sensors, " + label, fontsize=18)
        plt.xlabel('Time (ms)', fontsize=16)

        if gv.save_plots:
            fig_name = f"EEG_sensors_{label}.png"
            os.makedirs(gv.fig_folder, exist_ok=True)
            plt.savefig(os.path.join(gv.fig_folder, fig_name), dpi=200)
        if gv.show_plots:
            plt.show()

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

def ev(time_series_original):
    conn = get_conn()
    time_series = time_series_original.copy()
    tsr = TimeSeriesRegion(connectivity=conn,
                       data=time_series,                            #in TVB 4D format
                       sample_period=gv.temp_avg_period) #in ms
    tsr.configure()
    
    input_shape = tsr.data.shape
    result_shape = (input_shape[2], input_shape[2], input_shape[1], input_shape[3])
    result = np.zeros(result_shape)

    for mode in range(result_shape[3]):
        for var in range(result_shape[2]):
            data = tsr.data[:,var,:, mode].squeeze()
            result[:, :, var, mode] = np.corrcoef(data.T)

    corr_coeff = graph.CorrelationCoefficients(source=tsr, array_data=result)
    
    return corr_coeff

def functional_connectivity(time_series_original, label):

    fig_save_dir = os.path.join(gv.fig_folder, f"tau{gv.tau_stage}_abeta{gv.abeta_stage}_a{gv.aa:.4f}_b{gv.bb:.4f}_g{gv.coup:.4f}_v{gv.vel:.2f}_n{gv.noi:.4f}_simtime{gv.sim_time}")
    os.makedirs(fig_save_dir, exist_ok=True)

    conn = get_conn()
    time_series = time_series_original.copy()
    #time_series = time_series[np.isnan(time_series)]
    tsr_corr = ev(time_series)

    corr= tsr_corr.array_data[..., 0, 0]
    
    corr -= np.eye(len(time_series[0,0,:,0]))
    abs_corr_mean = np.mean(np.abs(corr))
    corr_mean = np.mean(corr)
    sub_corr = macro_fc(corr)
    abs_sub_corr_mean = np.mean(np.abs(sub_corr))
    corr_sub_mean = np.mean(sub_corr)

    #Define a threshold for connections
    corr_th = (corr) > gv.th

    pmatrix = np.zeros((len(time_series[0,0,:,0]),len(time_series[0,0,:,0])))

    for i in range(len(time_series[0,0,:,0])):
        for j in range(len(time_series[0,0,:,0])):
            
            region_A = time_series[:,0, i, 0]
            region_B = time_series[:,0, j, 0]
            
            pmatrix[i][j] = pearsonr(region_A, region_B)[1]

    pmatrix_th = pmatrix < 0.05

    corr_th_significance = np.multiply(corr_th, pmatrix_th)

    # FC data
    data_save_dir = os.path.join(gv.data_folder, f"tau{gv.tau_stage}_abeta{gv.abeta_stage}_a{gv.aa:.4f}_b{gv.bb:.4f}_g{gv.coup:.4f}_v{gv.vel:.2f}_n{gv.noi:.4f}_simtime{gv.sim_time}")
    os.makedirs(data_save_dir, exist_ok=True)

    
    with open(os.path.join(data_save_dir, f"FC_means_{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}.txt"), "w") as f:
        f.write(f"Label: {label}\n")
        f.write(f"Mean Correlation (raw): {corr_mean:.6f}\n")
        f.write(f"Mean Absolute Correlation: {abs_corr_mean:.6f}\n")
        f.write(f"Threshold (gv.th): {gv.th}\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    corr_viz = 2 * (corr - np.min(corr)) / (np.max(corr) - np.min(corr)) - 1
    # Plot corr_tavg
    im1 = ax1.imshow(corr_viz, interpolation='nearest', cmap='bwr')
    ax1.set_yticks(np.arange(len(conn.region_labels)))
    ax1.set_yticklabels(conn.region_labels, fontsize=12)
    ax1.set_xticks(np.arange(len(conn.region_labels)))
    ax1.set_xticklabels(conn.region_labels, rotation='vertical', fontsize=12)
    ax1.set_title(f"{label} FC mean = {corr_mean:.4f}, ({abs_corr_mean:.4f})", fontsize=15)

    # Plot corr_tavg_th
    im2 = ax2.imshow(corr_th, interpolation='nearest', cmap='bwr')
    ax2.set_yticks(np.arange(len(conn.region_labels)))
    ax2.set_yticklabels(conn.region_labels, fontsize=12)
    ax2.set_xticks(np.arange(len(conn.region_labels)))
    ax2.set_xticklabels(conn.region_labels, rotation='vertical', fontsize=12)
    ax2.set_title(label+' FC w/ threshold', fontsize=15)

    # Colorbar
    cax = fig.add_axes([ax2.get_position().x1 + 0.01, ax2.get_position().y0, 0.02, ax2.get_position().height])
    cbar = plt.colorbar(im2, cax=cax)

    if gv.save_plots:
        fig_name = f"FC_{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}.png"
        plt.savefig(os.path.join(fig_save_dir, fig_name), dpi=200)
    if gv.show_plots:
        plt.show()

    sub_corr_viz = 2 * (sub_corr - np.min(sub_corr)) / (np.max(sub_corr) - np.min(sub_corr)) - 1
    # Plot corr
    fig = plt.figure(figsize=(12, 8)) 
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
    im = ax.imshow(sub_corr_viz, cmap='bwr', vmin=-1, vmax=1, aspect='equal') 
    title_text = f"Reduced size simulated FC\n{label} FC reduced mean = {corr_sub_mean:.4f}, ({abs_sub_corr_mean:.4f})"
    ax.set_title(title_text, fontsize=15)
    cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
    plt.colorbar(im, cax=cax)

    output = os.path.join(data_save_dir, 
        f"FC_{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}.npz")

    np.savez_compressed(
            output,
            normal_size=corr,
            reduced_size=sub_corr
        )

    if gv.save_plots:
        fig_name = f"FC_reduced_{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}.png"
        plt.savefig(os.path.join(fig_save_dir, fig_name), dpi=200)
    if gv.show_plots:
        plt.show()


    

def functional_connectivity_band(time_series_original, label, lowcut, highcut):

    fig_save_dir = os.path.join(gv.fig_folder, f"tau{gv.tau_stage}_abeta{gv.abeta_stage}_a{gv.aa:.4f}_b{gv.bb:.4f}_g{gv.coup:.4f}_v{gv.vel:.2f}_n{gv.noi:.4f}_simtime{gv.sim_time}")
    os.makedirs(fig_save_dir, exist_ok=True)

    conn = get_conn()
    #time_series = time_series[np.isnan(time_series)]
    time_series = time_series_original.copy()

    for i in range(len(time_series[0,0,:,0])):
        time_series[:,0,i,0] = bandpass_filter(time_series[:,0,i,0], lowcut=lowcut, highcut=highcut)

    tsr_corr = ev(time_series)

    corr= tsr_corr.array_data[..., 0, 0]
    
    corr -= np.eye(len(time_series[0,0,:,0]))
    abs_corr_mean = np.mean(np.abs(corr))
    corr_mean = np.mean(corr)
    sub_corr = macro_fc(corr)
    abs_sub_corr_mean = np.mean(np.abs(sub_corr))
    corr_sub_mean = np.mean(sub_corr)

    #Define a threshold for connections
    corr_th = (corr) > gv.th

    pmatrix = np.zeros((len(time_series[0,0,:,0]),len(time_series[0,0,:,0])))

    for i in range(len(time_series[0,0,:,0])):
        for j in range(len(time_series[0,0,:,0])):
            
            region_A = time_series[:,0, i, 0]
            region_B = time_series[:,0, j, 0]
            
            pmatrix[i][j] = pearsonr(region_A, region_B)[1]

    pmatrix_th = pmatrix < 0.05

    corr_th_significance = np.multiply(corr_th, pmatrix_th)

    data_save_dir = os.path.join(gv.data_folder, f"tau{gv.tau_stage}_abeta{gv.abeta_stage}_a{gv.aa:.4f}_b{gv.bb:.4f}_g{gv.coup:.4f}_v{gv.vel:.2f}_n{gv.noi:.4f}_simtime{gv.sim_time}")
    os.makedirs(data_save_dir, exist_ok=True)
    
    with open(os.path.join(data_save_dir, f"FC_means_{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}_band_{lowcut}-{highcut}_Hz.txt"), "w") as f:
        f.write(f"Label: {label}\n")
        f.write(f"Mean Correlation (raw): {corr_mean:.6f}\n")
        f.write(f"Mean Absolute Correlation: {abs_corr_mean:.6f}\n")
        f.write(f"Threshold (gv.th): {gv.th}\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    corr_viz = 2 * (corr - np.min(corr)) / (np.max(corr) - np.min(corr)) - 1
    # Plot corr_tavg
    im1 = ax1.imshow(corr_viz, interpolation='nearest', cmap='bwr')
    ax1.set_yticks(np.arange(len(conn.region_labels)))
    ax1.set_yticklabels(conn.region_labels, fontsize=12)
    ax1.set_xticks(np.arange(len(conn.region_labels)))
    ax1.set_xticklabels(conn.region_labels, rotation='vertical', fontsize=12)
    ax1.set_title(f"{label} FC mean = {corr_mean:.4f}, ({abs_corr_mean:.4f})", fontsize=15)

    # Plot corr_tavg_th
    im2 = ax2.imshow(corr_th, interpolation='nearest', cmap='bwr')
    ax2.set_yticks(np.arange(len(conn.region_labels)))
    ax2.set_yticklabels(conn.region_labels, fontsize=12)
    ax2.set_xticks(np.arange(len(conn.region_labels)))
    ax2.set_xticklabels(conn.region_labels, rotation='vertical', fontsize=12)
    ax2.set_title(label+' FC w/ threshold', fontsize=15)

    # Colorbar
    cax = fig.add_axes([ax2.get_position().x1 + 0.01, ax2.get_position().y0, 0.02, ax2.get_position().height])
    cbar = plt.colorbar(im2, cax=cax)

    if gv.save_plots:
        fig_name = f"FC_matrix_{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}_band_{lowcut}-{highcut}_Hz.png"
        plt.savefig(os.path.join(fig_save_dir, fig_name), dpi=200)        
    if gv.show_plots:
        plt.show()

    sub_corr_viz = 2 * (sub_corr - np.min(sub_corr)) / (np.max(sub_corr) - np.min(sub_corr)) - 1
    # Plot corr
    fig = plt.figure(figsize=(12, 8)) 
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
    im = ax.imshow(sub_corr_viz, cmap='bwr', vmin=-1, vmax=1, aspect='equal') 
    title_text = f"Reduced size simulated FC\n{label} FC reduced mean = {corr_sub_mean:.4f}, ({abs_sub_corr_mean:.4f})"
    ax.set_title(title_text, fontsize=15)
    cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
    plt.colorbar(im, cax=cax)

    output = os.path.join(data_save_dir, 
        f"FC_{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}_band_{lowcut}-{highcut}_Hz.npz")

    np.savez_compressed(
            output,
            normal_size=corr,
            reduced_size=sub_corr,
            corr_th_sig = corr_th_significance,
            pmatrix = pmatrix
        )

    if gv.save_plots:
        fig_name = f"FC_reduced_{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}_band_{lowcut}-{highcut}_Hz.png"
        plt.savefig(os.path.join(fig_save_dir, fig_name), dpi=200)
    if gv.show_plots:
        plt.show()


#def functional_connectivity_alpha(time_series, label, lowcut, highcut):
    """
    Compute the alpha-band (8–13 Hz) functional connectivity matrix.

    This is a convenience wrapper around `functional_connectivity()`.

    Args:
        time_series (np.ndarray): Regional time series array.
        label (str): Dataset label.

    Returns:
        np.ndarray: Alpha-band functional connectivity matrix.
    """
"""
    time_series = bandpass_filter(time_series, lowcut=lowcut, highcut=highcut)

    n_timepoints = time_series.shape[0]
    n_regions = time_series.shape[2]
    filtered_signals = np.zeros((n_timepoints, n_regions))

    # Apply bandpass filter
    for i in range(n_regions):
        region_signal = time_series[:, 0, i, 0]
        filtered_signals[:, i] = bandpass_filter(region_signal, lowcut, highcut, order=4)

    # Compute PSD for each region
    psd_all = []
    for i in range(n_regions):
        freqs, psd_vals = signal.welch(
            filtered_signals[:, i],
            fs=1 / (gv.temp_avg_period / 1000),
            nperseg=min(n_seg, len(filtered_signals[:, i])),
            noverlap=n_seg // 2,
            window=window
        )
        psd_all.append(psd_vals)

    psd_all = np.array(psd_all)
    psd_mean = np.mean(psd_all, axis=0)
    psd_std = np.std(psd_all, axis=0)
    functional_connectivity(time_series, label)
"""

def make_train(node_idx, node_weights, **params):
    conn = get_conn()
    weighting = np.zeros(conn.weights.shape[0])
    weighting[node_idx] = node_weights
    eqn_t = equations.PulseTrain()
    eqn_t.parameters.update(params)
    stimulus = patterns.StimuliRegion(
        temporal=eqn_t,
        connectivity=conn,
        weight=weighting)
    return stimulus

def mean_channels(mask1, mask2, signal, chlist):
    summ = 0
    for ch in chlist:
        summ += signal[mask1:mask2, 0, ch, 0]
    summ /= len(chlist)
    return summ

def std_channels(mask1, mask2, signal, chlist):
    summ = 0
    mean = mean_channels(mask1, mask2, signal, chlist)
    for ch in chlist:
        summ += (signal[mask1:mask2, 0, ch, 0] - mean)**2
    #summ = sqrt(summ)
    summ /= len(chlist)-1
    return summ

def stimulus_visualization(visualizing, stimulus_onset, datas_stim, teeg_stim, mask1, mask2, chlist = [0, 10, 20, 30]):
    if visualizing == True:
        plt.figure(figsize=(10,10))
            
        for key, value in datas_stim.items():
            tavg_stim = value['tavg']
            eeg_stim = value['eeg']
            label = value['label']
            color = value['color']
            mean_n1 = mean_channels(mask1, mask2, eeg_stim, chlist)
            std_n1 =  std_channels(mask1, mask2, eeg_stim, chlist)
            plt.plot(teeg_stim[mask1:mask2], mean_n1, color = color, linewidth = 3, alpha = 0.55, label =  label)
            plt.fill_between(teeg_stim[mask1:mask2], mean_n1 + std_n1, mean_n1 - std_n1, color = color, linewidth = 3, alpha = 0.35 )
            print(label, 'N1 depth is:', np.min(mean_n1))

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.title("Simulated N1 component", fontsize = 29)
        plt.xlabel('Time (ms)', fontsize = 29)
        plt.ylabel(r'Voltage (10 $\mu$V)', fontsize = 29)
        plt.xticks([mask1, mask2], [mask1-stimulus_onset,mask2-stimulus_onset], fontsize = 22) #1500 is stimulus onset 
        plt.yticks(fontsize = 22)
        plt.legend(fontsize = 29)
        plt.show()

def simulate(sim_time=20000,
             tau_stage=0,
             abeta_stage=0,
             g=0.10004127233208432,
             velocity=10.374658598387436,
             noise=0.14480848489827142,
             a=0.0728044836955774,
             b=0.06293568322880852,
             factor=0.04,#0.03999999999999998,#0.8356062043764902, #1.7591351809125855e-03, #0.8581280031180116,
             C_ip_values=np.linspace(33.75, 10, 10),
             cip=0,
             C_ep_values=np.linspace(108, 20, 10),
             cep=0,
             stimulus=None):
    """
    Run a simulation of brain activity based on the Jansen–Rit neural mass model.

    Args:
        sim_time (float): Simulation duration in ms.
        structural_connectivities (np.ndarray): Structural connectivity matrices.
        noise (float): Noise level for stochastic integration.
        cp (int): Connectivity pattern index.
        np_parameter (int): Neuroplasticity parameter index.
        g (float): Global coupling strength.
        velocity (float): Signal conduction velocity.
        cip, cep, taue: Model parameters controlling inhibitory and excitatory coupling.
        tau_e_values, C_ip_values, C_ep_values: Parameter grids for tuning.
        impaired_regions (list): Brain regions to apply hypoinhibition.
        stimulus (object): Optional external stimulus object.

    Returns:
        tuple: (ttavg, tavg, teeg, eeg) — time vectors and simulated signals.
    """
    conn = get_conn()
    conn.weights = np.load(gv.struct_path)[tau_stage, :, :]
    conn.weights = conn.weights / 3 #np.max(conn.weights) #########################################################################################à
    conn.weights *= g
    conn.speed = np.array(velocity)
    conn.configure()

    # Model parameters (based on Jansen–Rit equations)
    Cep_value = C_ep_values[cep]
    Cip_value = C_ip_values[cip]
    J_value = 128
    J_stages = np.ones((76))
    tau_stage_1 = np.array([
        24, 62   # PHC
    ])

    tau_stage_2 = np.array([
        9, 47,       # Temporal allocortex / hippocampal limbic III -> HC
        31, 69,      # TCI
        30, 68,      # TCC
        34, 72       # TCV
    ])
    impaired_regions = np.array([21,22,30,31,32,34,59,60,68,69,70,72])
    tau_stage_3 = np.array([
        18, 56,   # PFCDL
        17, 55,   # PFCCL
        19, 57,   # PFCDM
        12, 50,   # M1
        28, 66,   # S1
        29, 67,   # S2
        35, 73,   # V1 
        36, 74    # V2
    ])
    tau_stage_all = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 20, 21, 22, 23, 25, 26, 27, 32, 33, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 51, 52, 53, 54, 58, 59, 60, 61, 63, 64, 65, 70, 71, 75
    ])

    degeneration = [0, 0, 0, 0.28, 0.78, 0.95, 0.95, 0.95]

    b_delta = 0.3133/0.0593

    a_array = np.ones((76,1))*a
    b_array = np.ones((76,1))*b*(1+b_delta*0.01*tau_stage)
    
    regions_centers = conn.centres
    distance = np.ones((76,1))
    wide_y = np.max(regions_centers[:,1]) - np.min(regions_centers[:,1])
    for i in range(len(regions_centers[:,1])):
        region_coordinate_y = regions_centers[i,1]
        distance[i] = (region_coordinate_y - np.max(regions_centers[:,1]))/wide_y
        a_array[i] = a*(1 + distance[i]*factor)
        #b_array[i] = b*(1 + distance[i]*factor)

    J_delta = 1 #0.0134/0.0192

    for i in tau_stage_1:
        J_stages[i] = 1 - degeneration[tau_stage + 2]*J_delta
        b_array[i] = b*(1+b_delta*0.5*degeneration[tau_stage + 2])
    
    for i in tau_stage_2:
        J_stages[i] = 1 - degeneration[tau_stage + 1]*J_delta
        b_array[i] = b*(1+b_delta*0.5*degeneration[tau_stage + 1])

    for i in tau_stage_3:
        J_stages[i] = 1 - degeneration[tau_stage]*J_delta
        b_array[i] = b*(1+b_delta*0.5*degeneration[tau_stage])
    #for i in impaired_regions:
    #    J_stages[i] = 1 - degeneration[tau_stage]*J_delta

    #for i in tau_stage_all:
    #    J_stages[i] = 1 - degeneration[tau_stage]*J_delta
    #a_array = np.load(gv.a_path)[abeta_stage, np.newaxis].T


    #b_array = np.load(gv.b_path)[abeta_stage, np.newaxis].T
    model_pars = dict(
        A=3.25, B=22, v0=6.0,
        a=a_array, b=b_array, r=0.56,
        nu_max=0.0025, J=J_value*J_stages,
        a_1=1.0, a_2=Cep_value / J_value,
        a_3=0.25, a_4=Cip_value / J_value,
        mu=0.22
    )

    # Add stochastic noise
    nsig = np.zeros((8, 76, 1))
    nsig[4] = model_pars['A'] * model_pars['a'] * (.320 - .120) * noise
    tvb_noise = tvb.noise.Additive(nsig=nsig)

    # Define monitors
    mon_eeg = get_eeg_monitor()
    mon_tavg = monitors.TemporalAverage(period=gv.temp_avg_period)

    sim = simulator.Simulator(
        connectivity=conn,
        conduction_speed=float(conn.speed),
        model=JRPSP(
            variables_of_interest=("y0", "y1", "y2", "y3", "y4", "y5", "y6", "y7"),
            **{k: np.array(v) for k, v in model_pars.items()}
        ),
        coupling=tvb.coupling.Linear(a=np.r_[a * 10]),
        integrator=tvb.integrators.EulerStochastic(dt=0.1, noise=tvb_noise),
        stimulus=stimulus,
        monitors=(mon_tavg, mon_eeg),
        simulation_length=sim_time
    )
    sim.configure()

    (ttavg, tavg), (teeg, eeg) = sim.run(simulation_length=sim_time)
    if gv.save_plots == True:
        # 1. Crea un nome univoco basato sui parametri chiave
        sim_id = f"tau{tau_stage}_abeta{abeta_stage}_a{a:.4f}_b{b:.4f}_g{g:.4f}_v{velocity:.2f}_n{noise:.4f}_simtime{sim_time}"

        data_save_dir = os.path.join(gv.data_folder, f"tau{gv.tau_stage}_abeta{gv.abeta_stage}_a{gv.aa:.4f}_b{gv.bb:.4f}_g{gv.coup:.4f}_v{gv.vel:.2f}_n{gv.noi:.4f}_simtime{gv.sim_time}")
        os.makedirs(data_save_dir, exist_ok=True)
        """
        output_file_name = f"times_and_eeg_{sim_id}.npz"
        output_file_path = os.path.join(data_save_dir, output_file_name)

        # 3. Salva i quattro array (Time Vector e Time Series) in un unico file compresso
        np.savez_compressed(
            output_file_path,
            teeg=teeg,  # Sarà accessibile come 'ttavg'
            eeg=eeg,    # Sarà accessibile come 'tavg'
        )

        output_file_name = f"times_and_tavg_{sim_id}.npz"
        output_file_path = os.path.join(data_save_dir, output_file_name)

        # 3. Salva i quattro array (Time Vector e Time Series) in un unico file compresso
        np.savez_compressed(
            output_file_path,
            ttavg=ttavg,  # Sarà accessibile come 'ttavg'
            tavg=tavg,    # Sarà accessibile come 'tavg'
        )
        """
            
    return ttavg, tavg, teeg, eeg


def psd(time_series_original, label, lowcut=0.5, highcut=45, n_seg=2048, window='hamming', return_psd=False):
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
    tau_stages = tau_stage_3+tau_stage_2+tau_stage_1

    psd_all = np.array(psd_all)
    psd_mean = np.mean(psd_all, axis=0)
    psd_std = np.std(psd_all, axis=0)

    # Normalize
    normalization_factor = np.max(psd_mean)
    psd_mean /= normalization_factor
    psd_std /= normalization_factor

    # psd_all array dimensions are [n_regions, n_freqs]
    if gv.save_plots == True:
        data_save_dir = os.path.join(gv.data_folder, f"tau{gv.tau_stage}_abeta{gv.abeta_stage}_a{gv.aa:.4f}_b{gv.bb:.4f}_g{gv.coup:.4f}_v{gv.vel:.2f}_n{gv.noi:.4f}_simtime{gv.sim_time}")
        os.makedirs(data_save_dir, exist_ok=True)
        fig_save_dir = os.path.join(gv.fig_folder, f"tau{gv.tau_stage}_abeta{gv.abeta_stage}_a{gv.aa:.4f}_b{gv.bb:.4f}_g{gv.coup:.4f}_v{gv.vel:.2f}_n{gv.noi:.4f}_simtime{gv.sim_time}")
        os.makedirs(fig_save_dir, exist_ok=True)

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
        if gv.save_plots == True:
            output_non_affected = os.path.join(data_save_dir, 
                f"PSD_NON_AFFECTED_{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}.npz")

            np.savez_compressed(
                output_non_affected,
                freqs=freqs_non_affected,
                psd_mean=psd_non_affected_mean,
                psd_std=psd_non_affected_std
            )

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

        # Save / show
        if gv.save_plots:
            fig_name = f"{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}_non_affected_PSD.png"
            plt.savefig(os.path.join(fig_save_dir, fig_name), dpi=200)
        if gv.show_plots:
            plt.show()
        else:
            plt.close()

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
        if gv.save_plots == True:
            output_affected = os.path.join(data_save_dir, 
                    f"PSD_AFFECTED_{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}.npz")

            np.savez_compressed(
                output_affected,
                freqs=freqs_affected,
                psd_mean=psd_affected_mean,
                psd_std=psd_affected_std
            )

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

        # Save / show
        if gv.save_plots:
            fig_name = f"{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}_affected_PSD.png"
            plt.savefig(os.path.join(fig_save_dir, fig_name), dpi=200)
        if gv.show_plots:
            plt.show()
        else:
            plt.close()
    
    if gv.save_plots:
        output_all = os.path.join(data_save_dir, 
            f"PSD_{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}.npz")

        np.savez_compressed(
            output_all,
            freqs=freqs,
            psd_mean=psd_mean,
            psd_std=psd_std
        )
        
        # normalization factor
        with open(os.path.join(data_save_dir, f"PSD_info_{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}.txt"), "w") as f:
            f.write(f"Label: {label}\n")
            f.write(f"Sampling Frequency (fs): {fs_welch:.2f} Hz\n")
            f.write(f"Normalization Factor (max mean PSD): {normalization_factor:.6e}\n")
            if len(filtered_signals[0, :]) > 75:  
                f.write(f"Normalization Factor affected regions (max mean PSD): {affected_normalization_factor:.6e}\n")
                f.write(f"Normalization Factor non affected regions (max mean PSD): {non_affected_normalization_factor:.6e}\n")
            f.write(f"Bandpass Filter: {lowcut}-{highcut} Hz\n")
            f.write(f"NPERSEG (Welch): {n_seg}\n")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, psd_mean, label=label, linewidth=3)
    plt.fill_between(freqs, psd_mean + psd_std, psd_mean - psd_std, alpha=0.35)
    plt.title(label + ' mean PSD', fontsize=20)
    plt.ylim(0, 1.01)
    plt.xlim(0, 45)
    plt.xlabel('Frequency [Hz]', fontsize=16)
    plt.ylabel('Normalized PSD', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)

    # Save / show
    if gv.save_plots:
        fig_name = f"{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}_PSD.png"
        os.makedirs(gv.fig_folder, exist_ok=True)
        plt.savefig(os.path.join(fig_save_dir, fig_name), dpi=200)
    if gv.show_plots:
        plt.show()
    else:
        plt.close()

    if return_psd:
        return freqs, psd_mean

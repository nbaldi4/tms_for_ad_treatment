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
    skin = surfaces.SkinAir.from_file(r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\sub-CLB00044_outer_skin.zip")
    skin.configure()

    mon_EEG = get_eeg_monitor()
    # Load and configure EEG sensor positions
    sens_eeg = mon_EEG.sensors
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

def brain_activity(eeg, tavg, end_time=3000):
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
    eeg = eeg[:end_time, :]
    eeg /= (np.max(eeg, 0) - np.min(eeg, 0))
    eeg -= np.mean(eeg, 0)
    times_eeg = np.arange(eeg.shape[0])

    # === Regional signals ===
    tavg = tavg[:end_time, :]
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

def eeg_plot(datas, times, display_time=19000, chlist=[4, 13, 20, 23, 38, 41, 45, 50, 62]):
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
        plt.plot(times[1000:1000+display_time],
                 time_series[1000:1000+display_time, 0, chlist, 0] + np.r_[:len(chlist)],
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

def bandpass_filter(data, lowcut, highcut, fs=1 / (gv.temp_avg_period / 1000), order=4):
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
            result[:, :, var, mode] = np.corrcoef(data.T) #pearson correlation

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


    

def functional_connectivity_band(time_series_original, label, lowcut, highcut, patient = 0, real_patient = False):

    if real_patient == False:
        fig_save_dir = os.path.join(gv.fig_folder, f"tau{gv.tau_stage}_abeta{gv.abeta_stage}_rs{gv.random_seed}") #_a{gv.aa:.4f}_b{gv.bb:.4f}_g{gv.coup:.4f}_v{gv.vel:.2f}_n{gv.noi:.4f}_simtime{gv.sim_time}_rs{gv.random_seed}") #gv.fig_folder
        os.makedirs(fig_save_dir, exist_ok=True)
    else:
        fig_save_dir = os.path.join(gv.fig_folder, f"patient_{patient}")
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

    if real_patient == False:
        data_save_dir = os.path.join(gv.data_folder, f"tau{gv.tau_stage}_abeta{gv.abeta_stage}_rs{gv.random_seed}") #_a{gv.aa:.4f}_b{gv.bb:.4f}_g{gv.coup:.4f}_v{gv.vel:.2f}_n{gv.noi:.4f}_simtime{gv.sim_time}_rs{gv.random_seed}")#gv.data_folder
        os.makedirs(data_save_dir, exist_ok=True)
        file_txt_title = f"FC_means_{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}_{lowcut}-{highcut}_Hz.txt"
    else:
        data_save_dir = os.path.join(gv.data_folder, f"patient_{patient}")
        os.makedirs(data_save_dir, exist_ok=True)
        file_txt_title = f"FC_patient_{patient}_{lowcut}-{highcut}_Hz.txt"
    
    with open(os.path.join(data_save_dir, file_txt_title), "w") as f:
        f.write(f"Label: {label}\n")
        f.write(f"Mean Correlation (raw): {corr_mean:.6f}\n")
        f.write(f"Mean Absolute Correlation: {abs_corr_mean:.6f}\n")
        f.write(f"Threshold (gv.th): {gv.th}\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    corr_viz = 2 * (corr - np.min(corr)) / (np.max(corr) - np.min(corr)) - 1
    # Plot corr_tavg
    im1 = ax1.imshow(corr_viz, interpolation='nearest', cmap='bwr')
    #ax1.set_yticks(np.arange(len(conn.region_labels)))
    #ax1.set_yticklabels(conn.region_labels, fontsize=12)
    #ax1.set_xticks(np.arange(len(conn.region_labels)))
    #ax1.set_xticklabels(conn.region_labels, rotation='vertical', fontsize=12)
    ax1.set_title(f"{label} FC mean = {corr_mean:.4f}, ({abs_corr_mean:.4f})", fontsize=15)

    # Plot corr_tavg_th
    im2 = ax2.imshow(corr_th, interpolation='nearest', cmap='bwr')
    #ax2.set_yticks(np.arange(len(conn.region_labels)))
    #ax2.set_yticklabels(conn.region_labels, fontsize=12)
    #ax2.set_xticks(np.arange(len(conn.region_labels)))
    #ax2.set_xticklabels(conn.region_labels, rotation='vertical', fontsize=12)
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

    if real_patient == False:
        output = os.path.join(data_save_dir, 
            f"FC_{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}_band_{lowcut}-{highcut}_Hz.npz")
    else:
        output = os.path.join(data_save_dir, 
            f"FC_patient_{patient}_band_{lowcut}-{highcut}_Hz.npz")

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
    else:
        plt.close()


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

def simulate(sim_time=20000,
             tau_stage=gv.tau_stage,
             abeta_stage=gv.abeta_stage,
             g=gv.coup, #0.10004127233208432,
             velocity=gv.vel, #10.374658598387436,
             noise=gv.noi, #0.14480848489827142,
             a=gv.aa, #0.0728044836955774,
             b=gv.bb, #0.06293568322880852,
             factor=gv.factor, #0.04,#0.03999999999999998,#0.8356062043764902, #1.7591351809125855e-03, #0.8581280031180116,
             C_ip_values=np.linspace(33.75, 13.25, 10),
             cip=0,
             C_ep_values=np.linspace(108, 28, 10),
             cep=0,
             stimulus=None, 
             save_raw=False):
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
    conn.weights = conn.weights / 3 #np.max(conn.weights)
    conn.weights *= g
    conn.speed = np.array(velocity)
    conn.configure()

    # Model parameters (based on Jansen–Rit equations)
    Cep_value = C_ep_values[cep]
    Cip_value = C_ip_values[cip]
    J_value = 128
    J_stages = np.ones((76,1))

    tau_stage_1 = [ #2 areas
        24, 62  # PHC (parahippocampal gyrus)
    ]

    tau_stage_2 = [ #16 areas
        2, 40,   # AMYG (amygdala)
        9, 47,   # HC (hippocampus)
        13, 51,  # PCI (inferor parietal cortex)
        30, 68,  # TCC (central temporal)
        31, 69,  # TCI (inferior temporal)
        32, 70,  # TCPOL (temporal pole)
        33, 71,  # TCS (superior temporal)
        34, 72   # TCV (ventral temporal)
    ]

    tau_stage_3 = [ #34 areas
        0, 38,   # A1 (primary uditivity cortex)
        11, 49,  # IP (intraparietal cortex)
        12, 50,  # M1 (primary motor cortex)
        14, 52,  # PCIP (posterior inferor parietal cortex)
        15, 53,  # PCM (medial parietal cortex)
        16, 54,  # PCS (parietal sulcus)
        17, 55,  # PFCCL (caudolateral prefrontal cortex)
        18, 56,  # PFCDL (dorsolateral prefrontal cortex)
        19, 57,  # PFCDM (dorsomedial prefrontal cortex)
        21, 59,  # PFCORB (orbitofrontal)
        22, 60,  # PFCPOL (prefrontal polar)
        23, 61,  # PFCVL (ventrolateral prefrontal cortex)
        25, 63,  # PMCDL (ventrolateral premotor cortex)
        26, 64,  # PMCM (medial premotor cortex)
        27, 65,  # PMCVL (ventrolaterl premotor cortex)
        28, 66,  # S1 (primary somatosensorial cortex)
        35, 73,  # V1 (primary visul cortex)
    ]

    Abeta_stage_1 = [ #12 areas
        21, 59,   # PFCORB (orbitofrontal)
        22, 60,   # PFCPOL (prefrontal polar)
        30, 68,   # TCC (central temporal)
        31, 69,   # TCI (inferior temporal)
        32, 70,   # TCPOL (temporal pole)
        34, 72    # TCV (ventral temporal)
    ]

    Abeta_stage_2 = [ #22 areas
        2, 40,   # AMYG (amygdala)
        3, 41,   # CCA (anterior cingulate cortex)
        4, 42,   # CCP (posterior cingulate cortex)
        5, 43,   # CCR (rostral cingulate cortex)
        6, 44,   # CCS (subgenual cingulate cortex)
        9, 47,   # HC (hippocampus)
        10, 48,  # IA (insula = anterior + posterior)
        23, 61,  # PFCVL (ventrolateral prefrontal cortex)
        24, 62,  # PHC (parahippocampal gyrus)
        33, 71,  # TCS (superior temporal)
        37, 75   # CC (cingulate cortex)
    ]

    Abeta_stage_3 = [ #42 areas
        0, 38,   # A1 (primary uditivity cortex)
        1, 39,   # A2 (secondary uditivity cortex)
        7, 45,   # FEF (frontal eyes field)
        8, 46,   # G (gyrus)
        11, 49,  # IP (intraparietal cortex)
        12, 50,  # M1 (primary motor cortex)
        13, 51,  # PCI (inferor parietal cortex)
        14, 52,  # PCIP (posterior inferor parietal cortex)
        15, 53,  # PCM (medial parietal cortex)
        16, 54,  # PCS (parietal sulcus)
        17, 55,  # PFCCL (caudolateral prefrontal cortex)
        18, 56,  # PFCDL (dorsolateral prefrontal cortex)
        19, 57,  # PFCDM (dorsomedial prefrontal cortex)
        20, 58,  # PFCM (medial prefrontal cortex)
        25, 63,  # PMCDL (ventrolateral premotor cortex)
        26, 64,  # PMCM (medial premotor cortex)
        27, 65,  # PMCVL (ventrolaterl premotor cortex)
        28, 66,  # S1 (primary somatosensorial cortex)
        29, 67,  # S2 (secondary somatosensorial cortex)
        35, 73,  # V1 (primary visul cortex)
        36, 74,  # V2 (secondary visual cortex)
    ]

    Abeta_stage_1_second_trial = [  #alternative abeta staging
        30, 58, #Precuneus (PCu)
        35, 73, #Superior Frontal
        31, 69, #Superior Parietal
        28, 66, #Posterior Cingulate
        37, 75, #Middle Temporal
        36,	74  #Superior Temporal
    ]
        
    Abeta_stage_2_second_trial = [
        12, 50, #Hippocampus
        6, 44, #Entorhinal Cortex
        1, 39, #Amygdala
        13, 51, #Insula	
        32, 2, 70, 40, #Anterior Cingulate (Ros/Caud)
	    22, 60 #Parahippocampal Gyrus
    ]

    Abeta_stage_3_second_trial = [
        34, 72, #Thalamus
    	3, 41, #Caudate
	    33, 71, #Putamen
	    21, 59, #Pallidum
	    0, 38, #Accumbens
        ]

    degeneration = [0, 0, 0, (1/(1+19**(1/3))), (1/(1+19**(-1/3))), 0.95, 0.95, 0.95] 
    #stages derving from sigma function division in 3 parts (0, x1=1/3, x2=2/3, 1) s.t. f(0) = 0.05, f(1) = 0.95
    #alternative stagings, different between abeta and tau
    degeneration_tau = [0, 0, 0, 0.28, 0.78, 0.95, 0.95, 0.95]
    degeneration_abeta = [0, 0, 0, 0.6116, 0.8664, 0.95, 0.95, 0.95]
    degeneration_alpha = [[0, 0, 0], [0.11, 0.21, 0.31], [0.43, 0.53, 0.63], [0.75, 0.85, 0.95], [0.85, 0.95, 0.95], [0.95, 0.95, 0.95]]
    degeneration_alpha_2 = [[0, 0, 0], [0.21, 0.26, 0.31], [0.53, 0.58, 0.63], [0.85, 0.90, 0.95], [0.90, 0.95, 0.95], [0.95, 0.95, 0.95]]

    a_delta = (2.20*3.65/3.25)-1   ########################### Attenzione, valore di M
    b_delta = 13.25/33.75-1           #0.03133/0.0593-1

    a_array = np.ones((76,1))
    b_array = np.ones((76,1))*b #*(1+b_delta*0.01*tau_stage)
    regions_centers = conn.centres
    distance = np.ones((76,1))
    wide_y = np.max(regions_centers[:,1]) - np.min(regions_centers[:,1])
    for i in range(len(regions_centers[:,1])):
        region_coordinate_y = regions_centers[i,1]
        distance[i] = (region_coordinate_y - np.max(regions_centers[:,1]))/wide_y
        a_array[i] = a*(1 + distance[i]*factor)
    J_delta = 0.3

    #print('a iniziale', a_array[21:31])
    #print('b iniziale', b_array[21:31])
    #print('J iniziale', J_stages[21:31])
    for i in tau_stage_1:
        J_stages[i] = 1 - degeneration[tau_stage + 2]*J_delta
        b_array[i] = min(b_array[i], b*(1+b_delta*(1.8/2.2)*degeneration[tau_stage + 2]))
    
    for i in tau_stage_2:
        J_stages[i] = 1 - degeneration[tau_stage + 1]*J_delta
        b_array[i] = min(b_array[i], b*(1+b_delta*(1.8/2.2)*degeneration[tau_stage + 1]))

    for i in tau_stage_3:
        J_stages[i] = 1 - degeneration[tau_stage]*J_delta
        b_array[i] = min(b_array[i], b*(1+b_delta*(1.8/2.2)*degeneration[tau_stage]))

    for i in Abeta_stage_1:
        a_array[i] = a_array[i]*(1+a_delta*degeneration[abeta_stage + 2])
        b_array[i] = b_array[i] + b*b_delta*(0.4/2.2)*degeneration[abeta_stage+ 2]

    for i in Abeta_stage_2:
        a_array[i] = a_array[i]*(1+a_delta*degeneration[abeta_stage + 1])
        b_array[i] = b_array[i] + b*b_delta*(0.4/2.2)*degeneration[abeta_stage + 1]

    for i in Abeta_stage_3:
        a_array[i] = a_array[i]*(1+a_delta*degeneration[abeta_stage])
        b_array[i] = b_array[i] + b*b_delta*(0.4/2.2)*degeneration[abeta_stage]
    #print('a finale', a_array[21:31])
    #print('b finale', b_array[21:31])
    #print('J finale', J_stages[21:31])
    model_pars = dict(
        A=3.25, B=22, v0=6.0,
        a=a_array, b=b_array, r=0.56,
        nu_max=0.0025, J=J_value*J_stages,
        a_1=1.0, a_2 = Cep_value / J_value, #a_2 around 0.84375
        a_3=0.25, a_4 = Cip_value / J_value, #a_4 around 0.26367
        mu=0.22
    )

    # Add stochastic noise
    nsig = np.zeros((8, 76, 1))
    nsig[4] = model_pars['A'] * model_pars['a'] * (.320 - .120) * noise
    tvb_noise = tvb.noise.Additive(nsig=nsig, noise_seed=gv.random_seed)

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
        coupling=tvb.coupling.Linear(a=np.r_[a_array * 10]),
        integrator=tvb.integrators.EulerStochastic(dt=0.1, noise=tvb_noise),
        stimulus=stimulus,
        monitors=(mon_tavg, mon_eeg),
        simulation_length=sim_time
    )
    sim.configure()

    (ttavg, tavg), (teeg, eeg) = sim.run(simulation_length=sim_time)
    if save_raw==True:
        data_save_dir = os.path.join(gv.data_folder, f"tau{gv.tau_stage}_abeta{gv.abeta_stage}_rnd{gv.random_seed}")
        os.makedirs(data_save_dir, exist_ok=True)
        output = os.path.join(data_save_dir, f"eeg_and_teeg_raw_tau{gv.tau_stage}_abeta{gv.abeta_stage}_rnd{gv.random_seed}.npz")
        metadata = {
            "description": "File contains eeg (and time eeg) files and tavg (and time tavg) files"
        }

        np.savez_compressed(
            output,
            eeg=eeg,
            teeg=teeg,
            tavg=tavg,
            ttavg=ttavg,
            metadata=metadata
        )
            
    return ttavg, tavg, teeg, eeg


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
    """
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

    """

    tau_stage_1 = [ # 4 regions
        4, 40, 47, 53
    ]

    tau_stage_2 = [ # 17 regions
        7, 13, 14, 16, 18, 31, 37, 38, 39, 41, 45, 65, 66, 67, 80, 84, 85
    ]

    tau_stage_3 = [ # 38 regions
        0, 2, 3, 6, 9, 11, 15, 17, 19, 21, 22, 23, 24, 25, 27, 29, 33, 34, 44, 46, 49, 51, 52,
        55, 58, 62, 64, 68, 70, 71, 72, 73, 74, 75, 76, 81, 82, 83
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

    """
    print(psd_mean)
    print(freqs[(freqs >= 0.5) & (freqs < 4)])
    print(freqs[(freqs >= 4) & (freqs < 8)])
    print(freqs[(freqs >= 8) & (freqs < 13)])
    print(freqs[(freqs >= 13) & (freqs < 30)])
    print(freqs[(freqs >= 0.5) & (freqs < 45)])
    print(np.max(psd_mean))
    """

    # psd_all array dimensions are [n_regions, n_freqs]
    
    if real_patient == False:
        fig_save_dir = os.path.join(gv.fig_folder, f"tau{gv.tau_stage}_abeta{gv.abeta_stage}_rs{gv.random_seed}") #_a{gv.aa:.4f}_b{gv.bb:.4f}_g{gv.coup:.4f}_v{gv.vel:.2f}_n{gv.noi:.4f}_simtime{gv.sim_time}_rs{gv.random_seed}") #gv.fig_folder
        os.makedirs(fig_save_dir, exist_ok=True)
        data_save_dir = os.path.join(gv.data_folder, f"tau{gv.tau_stage}_abeta{gv.abeta_stage}_rs{gv.random_seed}") #_a{gv.aa:.4f}_b{gv.bb:.4f}_g{gv.coup:.4f}_v{gv.vel:.2f}_n{gv.noi:.4f}_simtime{gv.sim_time}_rs{gv.random_seed}")  #gv.data_folder
        os.makedirs(data_save_dir, exist_ok=True)
    else:
        fig_save_dir = os.path.join(gv.fig_folder, f"patient_{patient}")
        os.makedirs(fig_save_dir, exist_ok=True)
        data_save_dir = os.path.join(gv.data_folder, f"patient_{patient}")
        os.makedirs(data_save_dir, exist_ok=True)


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

        if gv.save_plots == True:
            if real_patient == False:
                output_non_affected = os.path.join(data_save_dir, 
                    f"PSD_NON_AFFECTED_{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}.npz")
            else:
                output_non_affected = os.path.join(data_save_dir, 
                    f"PSD_NON_AFFECTED_patient_{patient}.npz")


            np.savez_compressed(
                output_non_affected,
                freqs=freqs_non_affected,
                psd_mean=psd_non_affected_mean,
                psd_std=psd_non_affected_std
            )

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
        
        if gv.save_plots == True:
            if real_patient == False:
                output_affected = os.path.join(data_save_dir, 
                    f"PSD_AFFECTED_{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}.npz")
            else:
                output_affected = os.path.join(data_save_dir, 
                    f"PSD_AFFECTED_{label}_patient_{patient}.npz")


            np.savez_compressed(
                output_affected,
                freqs=freqs_affected,
                psd_mean=psd_affected_mean,
                psd_std=psd_affected_std
            )

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

        # Save / show
        if gv.save_plots:
            if real_patient == False:
                fig_name = f"{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}_affected_PSD.png"
            else:
                fig_name = f"{label}_patient_{patient}_affected_PSD.png"
            plt.savefig(os.path.join(fig_save_dir, fig_name), dpi=200)
        if gv.show_plots:
            plt.show()
        else:
            plt.close()

    if gv.save_plots == True:
        if real_patient == False:
            output_all = os.path.join(data_save_dir, 
                f"PSD_{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}.npz")
        else:
            output_all = os.path.join(data_save_dir, 
                f"PSD_{label}_patient_{patient}.npz")

        np.savez_compressed(
            output_all,
            freqs=freqs,
            psd_all_channels=psd_all,
            psd_mean=psd_mean,
            psd_std=psd_std
        )
    
        # Normalize
        normalization_factor = np.max(psd_mean)
        #psd_mean /= normalization_factor
        #psd_std /= normalization_factor

        # normalization factor
        if real_patient == False:
            file_name = f"PSD_info_{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}.txt"
        else:
            file_name = f"PSD_patient_{patient}_info_{label}.txt"
        with open(os.path.join(data_save_dir, file_name), "w") as f:
            f.write(f"Label: {label}\n")
            f.write(f"Sampling Frequency (fs): {fs_welch:.2f} Hz\n")
            f.write(f"Normalization Factor (max mean PSD): {normalization_factor:.6e}\n")
            if len(filtered_signals[0, :]) > 75:  
                f.write(f"Normalization Factor affected regions (max mean PSD): {affected_normalization_factor:.6e}\n")
                f.write(f"Normalization Factor non affected regions (max mean PSD): {non_affected_normalization_factor:.6e}\n")
            f.write(f"Bandpass Filter: {lowcut}-{highcut} Hz\n")
            f.write(f"NPERSEG (Welch): {n_seg}\n")

    # Normalize
    normalization_factor = np.max(psd_mean)
    psd_mean_normalized = psd_mean.copy()
    psd_mean_normalized /= normalization_factor
    #psd_std /= normalization_factor

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, psd_mean_normalized, label=label, linewidth=3)
    #plt.fill_between(freqs, psd_mean + psd_std, psd_mean - psd_std, alpha=0.35)
    plt.title(label + ' mean PSD', fontsize=20)
    plt.ylim(0, 1.01)
    plt.xlim(0, 45)
    plt.xlabel('Frequency [Hz]', fontsize=16)
    plt.ylabel('PSD', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)

    # Save / show
    if gv.save_plots:
        if real_patient == False:
            fig_name = f"{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}_PSD.png"
        else:
            fig_name = f"PSD_patient_{patient}_{label}.png"
        os.makedirs(gv.fig_folder, exist_ok=True)
        plt.savefig(os.path.join(fig_save_dir, fig_name), dpi=200)
    if gv.show_plots:
        plt.show()
    else:
        plt.close()

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
        os.makedirs(data_save_dir, exist_ok=True)
        fig_name = f"{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}_interested_PSD.png"

        #if real_patient == False:
        #    fig_save_dir = os.path.join(gv.fig_folder, f"tau{gv.tau_stage}_abeta{gv.abeta_stage}_a{gv.aa:.4f}_b{gv.bb:.4f}_g{gv.coup:.4f}_v{gv.vel:.2f}_n{gv.noi:.4f}_simtime{gv.sim_time}_rs{gv.random_seed}") #gv.fig_folder
        #    os.makedirs(fig_save_dir, exist_ok=True)
        #    data_save_dir = os.path.join(gv.data_folder, f"tau{gv.tau_stage}_abeta{gv.abeta_stage}_a{gv.aa:.4f}_b{gv.bb:.4f}_g{gv.coup:.4f}_v{gv.vel:.2f}_n{gv.noi:.4f}_simtime{gv.sim_time}_rs{gv.random_seed}")  #gv.data_folder
        #    os.makedirs(data_save_dir, exist_ok=True)

        if gv.save_plots == True:
            if real_patient == False:
                output_AD = os.path.join(data_save_dir, 
                    f"PSD_{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}_interested.npz")
            else:
                output_AD = os.path.join(data_save_dir, 
                    f"PSD_{label}_patient_{patient}_interested.npz")
            np.savez_compressed(
                output_AD,
                freqs=freqs_AD,
                psd_mean=psd_interested_mean,
                psd_std=psd_interested_std
            )

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

        # Save / show
        if gv.save_plots:
            if real_patient == False:
                fig_name = f"{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}_AD_regions_PSD.png"
            else:
                fig_name = f"PSD_patient_{patient}_AD_regions_PSD_{label}.png"
            os.makedirs(gv.fig_folder, exist_ok=True)
            plt.savefig(os.path.join(fig_save_dir, fig_name), dpi=200)
        if gv.show_plots:
            plt.show()
        else:
            plt.close()

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
            
            f, Pxy = csd(data[:, i], data[:, j], fs=fs, nperseg=int(fs*2), noverlap=int(fs*2)//2) # finestre di 2s
            f, Pxx = welch(data[:, i], fs=fs, nperseg=int(fs*2), noverlap=int(fs*2)//2)
            f, Pyy = welch(data[:, j], fs=fs, nperseg=int(fs*2), noverlap=int(fs*2)//2)
            
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
    # FC data
    if real_patient == False:
        data_save_dir = os.path.join(gv.data_folder, f"tau{gv.tau_stage}_abeta{gv.abeta_stage}_a{gv.aa:.4f}_b{gv.bb:.4f}_g{gv.coup:.4f}_v{gv.vel:.2f}_n{gv.noi:.4f}_simtime{gv.sim_time}_rs{gv.random_seed}")#gv.data_folder
        os.makedirs(data_save_dir, exist_ok=True)
        file_txt_title = f"FC_means_{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}_{lowcut}-{highcut}_Hz.txt"
    else:
        data_save_dir = os.path.join(gv.data_folder, f"patient_{patient}")
        os.makedirs(data_save_dir, exist_ok=True)
        file_txt_title = f"FC_patient_{patient}_{lowcut}-{highcut}_Hz.txt"

    with open(os.path.join(data_save_dir, file_txt_title), "w") as f:
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

    if gv.save_plots:
        fig_name = f"FC_{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}_{lowcut}-{highcut}_Hz.png"
        plt.savefig(os.path.join(fig_save_dir, fig_name), dpi=200)
    if gv.show_plots:
        plt.show()    
    else:
        plt.close()

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

    if real_patient == False:
        output = os.path.join(data_save_dir, 
        f"FC_{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}_{lowcut}-{highcut}_Hz.npz")
    else:
        output = os.path.join(data_save_dir, 
        f"FC_{lowcut}-{highcut}_Hz_patient_{patient}.npz")

    np.savez_compressed(
            output,
            normal_size=corr,
            reduced_size=sub_corr
        )
    #print('regular size correlation: ', np.mean(np.abs(corr)))
    #print('reduced size correlation: ', np.mean(np.abs(sub_corr)))

    if gv.save_plots:
        fig_name = f"FC_reduced_{label}_tau_{gv.tau_stage}_abeta_{gv.abeta_stage}_{lowcut}-{highcut}_Hz.png"
        plt.savefig(os.path.join(fig_save_dir, fig_name), dpi=200)
    if gv.show_plots:
        plt.show()
    else:
        plt.close()

def simulate_different_combination_parameters(
             sim_time=20000,
             tau_stage=np.zeros(3), #b, J, weights
             abeta_stage=np.zeros(2), #a, b
             g=0.10004127233208432,
             velocity=10.374658598387436,
             noise=0.14480848489827142,
             a=0.0728044836955774,
             b=0.06293568322880852,
             factor=0, #0.04,#0.03999999999999998,#0.8356062043764902, #1.7591351809125855e-03, #0.8581280031180116,
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
    conn.weights = np.load(gv.struct_path)[tau_stage[2], :, :]
    conn.weights = conn.weights / 3 #np.max(conn.weights) #########################################################################################
    conn.weights *= g
    conn.speed = np.array(velocity)
    conn.configure()

    # Model parameters (based on Jansen–Rit equations)
    Cep_value = C_ep_values[cep]
    Cip_value = C_ip_values[cip]
    J_value = 128
    J_stages = np.ones((76))

    tau_stage_1 = np.array([ #2 areas
        24, 62   # PHC
    ])

    tau_stage_2 = np.array([ #14 areas
        2, 40,       # AMYG
        9, 47,       # HC
        30, 68,      # TCC
        31, 69,      # TCI
        32, 70,      # TCPOL
        33, 71,      # TCS
        34, 72       # TCV
    ])

    tau_stage_3 = np.array([ #42 areas
        0, 38,    # A1
        1, 39,    # A2
        11, 49,   # IP
        12, 50,   # M1
        13, 51,   # PCI
        14, 52,   # PCIP
        15, 53,   # PCM
        16, 54,   # PCS
        17, 55,   # PFCCL
        18, 56,   # PFCDL
        19, 57,   # PFCDM
        21, 59,   # PFCORB
        22, 60,   # PFCPOL
        23, 61,   # PFCVL
        25, 63,   # PMCDL
        26, 64,   # PMCM
        27, 65,   # PMCVL
        28, 66,   # S1
        29, 67,   # S2
        35, 73,   # V1
        36, 74    # V2
    ])

    Abeta_stage_1_second_trial = [
        30, 58, #Precuneus (PCu)
        35, 73, #Superior Frontal
        31, 69, #Superior Parietal
        28, 66, #Posterior Cingulate
        37, 75, #Middle Temporal
        36,	74, #Superior Temporal
    ]

    Abeta_stage_2_second_trial = [
        12, 50, #Hippocampus
        6, 44, #Entorhinal Cortex
        1, 39, #Amygdala
        13, 51, #Insula	
        32, 2, 70, 40, #Anterior Cingulate (Ros/Caud)
	    22, 60 #Parahippocampal Gyrus
    ]

    Abeta_stage_3_second_trial = [
        34, 72, #Thalamus
    	3, 41, #Caudate
	    33, 71, #Putamen
	    21, 59, #Pallidum
	    0, 38, #Accumbens
    ]

    Abeta_stage_1 = [ #16 areas
        21, 59,   # PFCORB (orbitofrontal)
        22, 60,   # PFCPOL (prefrontal polar)
        23, 61,   # Ventrolateral prefrontal cortex (PFCVL)
        30, 68,   # TCC (central temporal)
        31, 69,   # TCI (inferior temporal)
        32, 70,   # TCPOL (temporal pole)
        33, 71,   # TCS
        34, 72    # TCV (ventral temporal)
    ]

    Abeta_stage_2 = [ #18 areas
        2, 40,      # Amygdala (AMYG)
        3, 41,      # CCA
        4, 42,      # CCP
        5, 43,      # CCR
        6, 44,      # CCS
        9, 47,      # Hippocampus (HC)
        10, 48,     # Insula (IA) = anterior + posterior
        24, 62,     # Parahippocampal gyrus (PHC)
        37, 75      # CC
    ]

    Abeta_stage_3 = [ #38 areas
        0, 38,  #A1
        1, 39,  #A2
        11, 49, #IP
        12, 50, #M1
        13, 51, #PCI
        14, 52, #PCIP
        15, 53, #PCM
        16, 54, #PCS
        17, 55, #PFCCL
        18, 56, #PFCDL
        19, 57, #PFCDM
        20, 58, #PFCM
        25, 63, #PMCDL
        26, 64, #PMCM
        27, 65, #PMCVL
        28, 66, #S1
        29, 67, #S2
        35, 73, #V1
        36, 74, #V2
    ]

    degeneration = [0, 0, 0, 0.28, 0.78, 0.95, 0.95, 0.95]
    degeneration_2_tau = [0, 0, 0, 0.28, 0.78, 0.95, 0.95, 0.95]
    degeneration_2_abeta = [0, 0, 0, 0.6116, 0.8664, 0.95, 0.95, 0.95]
    degeneration_alpha = [[0, 0, 0], [0.11, 0.21, 0.31], [0.43, 0.53, 0.63], [0.75, 0.85, 0.95], [0.85, 0.95, 0.95], [0.95, 0.95, 0.95]]
    degeneration_alpha_2 = [[0, 0, 0], [0.21, 0.26, 0.31], [0.53, 0.58, 0.63], [0.85, 0.90, 0.95], [0.90, 0.95, 0.95], [0.95, 0.95, 0.95]]

    a_delta = 3.65/3.25-1
    b_delta = 0.3133/0.0593-1

    a_array = np.ones((76,1))*a ########################################################################
    b_array = np.ones((76,1))*b*(1+b_delta*0.01*tau_stage[0])
    
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
        J_stages[i] = 1 - degeneration_2_tau[tau_stage[1] + 2]*J_delta
        b_array[i] = b*(1+b_delta*0.5*degeneration_2_tau[tau_stage[0] + 2])
    
    for i in tau_stage_2:
        J_stages[i] = 1 - degeneration_2_tau[tau_stage[1] + 1]*J_delta
        b_array[i] = b*(1+b_delta*0.5*degeneration_2_tau[tau_stage[0] + 1])

    for i in tau_stage_3:
        J_stages[i] = 1 - degeneration_2_tau[tau_stage[1]]*J_delta
        b_array[i] = b*(1+b_delta*0.5*degeneration_2_tau[tau_stage[0]])

    for i in Abeta_stage_1:
        a_array[i] = a*(1+a_delta*degeneration_2_abeta[abeta_stage[0] + 2])
        b_array[i] = b_array[i] + b*b_delta*0.5*degeneration_2_abeta[abeta_stage[1]+ 2]

    for i in Abeta_stage_2:
        a_array[i] = a*(1+a_delta*degeneration_2_abeta[abeta_stage[0] + 1])
        b_array[i] = b_array[i] + b*b_delta*0.5*degeneration_2_abeta[abeta_stage[1] + 1]

    for i in Abeta_stage_3:
        a_array[i] = a*(1+a_delta*degeneration_2_abeta[abeta_stage[0]])
        b_array[i] = b_array[i] + b*b_delta*0.5*degeneration_2_abeta[abeta_stage[1]]

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
        sim_id = f"a{abeta_stage[0]}_b_abeta{abeta_stage[1]}_tau{tau_stage[0]}_J{tau_stage[1]}_weights{tau_stage[2]}"

        data_save_dir = os.path.join(gv.data_folder, f"a{abeta_stage[0]}_b_abeta{abeta_stage[1]}_tau{tau_stage[0]}_J{tau_stage[1]}_weights{tau_stage[2]}")
        os.makedirs(data_save_dir, exist_ok=True)
        gv.fig_folder = os.path.join(gv.fig_folder, f"a{abeta_stage[0]}_b_abeta{abeta_stage[1]}_tau{tau_stage[0]}_J{tau_stage[1]}_weights{tau_stage[2]}")
        gv.data_folder = os.path.join(gv.data_folder, f"a{abeta_stage[0]}_b_abeta{abeta_stage[1]}_tau{tau_stage[0]}_J{tau_stage[1]}_weights{tau_stage[2]}")
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

def Simulate2(g=1, velocity=np.inf, noise=50e-3, stimulus = None, sim_time = 20000):
    '''This function simulates signals at different levels of neurodegeneration parameters. The regions in which hypoinhibition is modelled are passed 
    through the "regions" argument which by default is equal to the "impaired_regions" listed above''' 

    con = get_conn() 
    con.weights = np.load(gv.struct_path)[0, :, :] #This alters the connectome according to equation 3 for a combination of cp and np
    # cp is comprised between o and 2 with 50 equal steps, while neuroplasticity is uniformally distributed between 0 and 2 with steps of 0.25 amplitude (plus a value of 2.5 at the end)
    con.weights = con.weights / np.max(con.weights)              #normalised by maximum weight  
    con.weights *= g
    con.speed = np.array(velocity)                               #To select conduction velocity of the signal
    con.configure()
    
    a_value = 0.1 #this determines the value of t_e
    #altering inhibition in impaired regions:
    b_regions = np.ones(76)*0.05 #default value

    # setup model based on paper's parameters
    model_pars = dict(
        A=3.25, #excitatory PSP
        B=22, #inhibitory PSP
        v0=6.0,
        a=a_value, # increased with respect to the healthy case. (also remember that TVB uses ms, not s)
        b=b_regions,    # decreased with respect to the healthy case. (also remember that TVB uses ms, not s)
        r=0.56,
        nu_max=0.0025, # e0 in the JR original paper
        # TVB factors C_i into J*a_i, e.g. C1 = a_1 * J
        J=128,
        a_1=1.0,
        a_2=0.8,
        a_3= 0.25,
        a_4= 0.25,
        mu=0.22, # baseline input, avg of 120 and 320 pulses/s
    )

    # factor out noise from dy4 = A a (p(t) + ...) as y4 += dt (...) + A a dW_t
    # this allows us to model the noise as TVB does it, though scaling requires experiment
    nsig = np.zeros((8, 76, 1))
    nsig[4] = model_pars['A'] * model_pars['a'] * (.320 - .120) * noise
    noise = tvb.noise.Additive(nsig=nsig)

    #setting up monitors:
    period = 1. #sampling time in terms of simulation time (total number of timepoints = sim_time / period)
    mon_tavg = monitors.TemporalAverage(period=period)
    mon_EEG = get_eeg_monitor()
    #Bundling
    what_to_watch = (mon_tavg, mon_EEG)
    

    sim = simulator.Simulator(connectivity=con,
                              conduction_speed=float(con.speed),
                              model=JRPSP(
                              variables_of_interest=("y0", "y1", "y2", "y3", "y4", "y5", "y6", "y7"),
                              **{k: np.array(v) for k, v in model_pars.items()}),
                              coupling=tvb.coupling.Linear(a=np.r_[a_value*10]),
                              integrator=tvb.integrators.EulerStochastic(dt=0.1, noise=noise),
                              monitors=what_to_watch,
                              stimulus = stimulus,
                              simulation_length=sim_time)
    sim.configure()

    (ttavg, tavg), (teeg, eeg) = sim.run(simulation_length=sim_time) #tavg is the name of signals in region space

    return ttavg, tavg, teeg, eeg # ttavg and teeg are the timestamps

def simulate_parametri_lorenzo(sim_time=20000,
             g=1, #0.10004127233208432,
             velocity=gv.vel, #10.374658598387436,
             noise=50e-3, #0.14480848489827142,
             tau_stage=gv.tau_stage,
             abeta_stage=gv.abeta_stage,
             a=gv.aa,
             b=gv.bb,
             cp = 0,
             lp = 0,
             np_parameter = 0,
             stimulus = None, 
             factor = 0
            ):
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
    conn.weights =  np.load(gv.struct_path)[0, :, :]
    conn.weights = conn.weights / 3 #########################################################################################
    conn.weights *= g
    conn.speed = np.array(np.inf)
    conn.configure()
    #impaired_regions = [21,22,30,31,32,34,59,60,68,69,70,72]
    # Model parameters (based on Jansen–Rit equations)
    #n_steps = 50 #How many steps you want to use for changing t_e and t_i
    #a_array = np.linspace(0.100,0.112,n_steps) #This is the array of considered t_e values, TVB uses'a' as a parameter which is 1/t_e
    #b_array = np.linspace(0.05,0.025,n_steps) #This is the array of considered t_i values, TVB uses 'b' as a parameter which is 1/t_i
    #a_value = a_array[lp] #this determines the value of t_e
    #b_value = b_array[lp] #this determines the value of t_i

        # Model parameters (based on Jansen–Rit equations)
    Cpi_value = 33.75
    J_value = 128
    J_stages = np.ones((76))
    tau_stage_1 = np.array([ #2 areas
        24, 62   # PHC
    ])

    tau_stage_2 = np.array([ #8 areas
        9, 47,       # Temporal allocortex / hippocampal limbic III -> HC
        31, 69,      # TCI
        30, 68,      # TCC
        34, 72       # TCV
    ])
    impaired_regions = np.array([21,22,30,31,32,34,59,60,68,69,70,72])
    tau_stage_3 = np.array([ #16 areas
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

    Abeta_stage_1 = [ #12 areas
        22, 60,   # PFCPOL (prefrontal polar)
        21, 59,   # PFCORB (orbitofrontal)
        31, 69,   # TCI (inferior temporal)
        30, 68,   # TCC (central temporal)
        34, 72,   # TCV (ventral temporal)
        32, 70    # TCPOL (temporal pole)
    ]

    Abeta_stage_1_second_trial = [
        30, 58, #Precuneus (PCu)
        35, 73, #Superior Frontal
        31, 69, #Superior Parietal
        28, 66, #Posterior Cingulate
        37, 75, #Middle Temporal
        36,	74, #Superior Temporal
    ]

    Abeta_stage_2 = [ #14 areas
        9, 47,      # Hippocampus (HC)
        24, 62,     # Parahippocampal gyrus (PHC)
        2, 40,      # Amygdala (AMYG)
        10, 48,     # Insula (IA) = anterior + posterior
        20, 58,     # Subgenual cingulum (PFCM)
        5, 43,      # Retrosplenial cingulum (CCR)
        23, 61      # Ventrolateral prefrontal cortex (PFCVL)
    ]

    Abeta_stage_2_second_trial = [
        12, 50, #Hippocampus
        6, 44, #Entorhinal Cortex
        1, 39, #Amygdala
        13, 51, #Insula	
        32, 2, 70, 40, #Anterior Cingulate (Ros/Caud)
	    22, 60 #Parahippocampal Gyrus
    ]

    Abeta_stage_3 = [ #50 areas
        0,  1,
        3,  4,
        6,  7,
        8, 11,
        12, 13,
        14, 15,
        16, 17,
        18, 19,
        25, 26,
        27, 28,
        29, 33,
        35, 36,
        37, 38,
        39, 41,
        42, 44,
        45, 46,
        49, 50,
        51, 52,
        53, 54,
        55, 56,
        57, 63,
        64, 65,
        66, 67,
        71, 73,
        74, 75
    ]

    Abeta_stage_3_second_trial = [
        34, 72, #Thalamus
    	3, 41, #Caudate
	    33, 71, #Putamen
	    21, 59, #Pallidum
	    0, 38, #Accumbens
    ]

    degeneration = [0, 0, 0, 0.28, 0.78, 0.95, 0.95, 0.95] ##########################################################################à
    degeneration_2_tau = [0, 0, 0, 0.28, 0.78, 0.95, 0.95, 0.95] ########################################################################
    degeneration_2_abeta = [0, 0, 0, 0.6116, 0.8664, 0.95, 0.95, 0.95] ########################################################################
    degeneration_alpha = [[0, 0, 0], [0.11, 0.21, 0.31], [0.43, 0.53, 0.63], [0.75, 0.85, 0.95], [0.85, 0.95, 0.95], [0.95, 0.95, 0.95]]
    degeneration_alpha_2 = [[0, 0, 0], [0.21, 0.26, 0.31], [0.53, 0.58, 0.63], [0.85, 0.90, 0.95], [0.90, 0.95, 0.95], [0.95, 0.95, 0.95]]
    
    a_delta = 3.65/3.25-1
    b_delta = 0.03133/0.0593-1#13.25/33.75-1 #0.3133/0.0593-1

    a_array = np.ones((76,1))
    b_array = np.ones((76,1))*b*(1+b_delta*0.01*tau_stage)
    
    regions_centers = conn.centres
    distance = np.ones((76,1))
    wide_y = np.max(regions_centers[:,1]) - np.min(regions_centers[:,1])
    for i in range(len(regions_centers[:,1])):
        region_coordinate_y = regions_centers[i,1]
        distance[i] = (region_coordinate_y - np.max(regions_centers[:,1]))/wide_y
        a_array[i] = a*(1 + distance[i]*factor)
        #b_array[i] = b*(1 + distance[i]*factor)
    
    J_delta = 0.3#1-0.0134/0.0191
    
    for i in tau_stage_1:
        J_stages[i] = 1 - degeneration[tau_stage + 2]*J_delta
        b_array[i] = b*(1+b_delta*0.5*degeneration[tau_stage + 2])
    
    for i in tau_stage_2:
        J_stages[i] = 1 - degeneration[tau_stage + 1]*J_delta
        b_array[i] = b*(1+b_delta*0.5*degeneration[tau_stage + 1])

    for i in tau_stage_3:
        J_stages[i] = 1 - degeneration[tau_stage]*J_delta
        b_array[i] = b*(1+b_delta*0.5*degeneration[tau_stage])

    for i in Abeta_stage_1:
        a_array[i] = a_array[i]*(1+a_delta*degeneration[abeta_stage + 2])
        b_array[i] = b_array[i] + b*b_delta*0.5*degeneration[abeta_stage+ 2]

    for i in Abeta_stage_2:
        a_array[i] = a_array[i]*(1+a_delta*degeneration[abeta_stage + 1])
        b_array[i] = b_array[i] + b*b_delta*0.5*degeneration[abeta_stage + 1]

    for i in Abeta_stage_3:
        a_array[i] = a_array[i]*(1+a_delta*degeneration[abeta_stage])
        b_array[i] = b_array[i] + b*b_delta*0.5*degeneration[abeta_stage]
    
    #altering inhibition in impaired regions:
    #b_regions = np.ones(76)*0.05 #default value
    #b_regions [impaired_regions]  = np.ones(len(impaired_regions)) * b_value # impairs b value (and therefore t_i) only in selected regions
    model_pars = dict(
        A=3.25, 
        B=22, 
        v0=6.0,
        a=0.1, 
        b=0.05,
        r=0.56,
        nu_max=0.0025, 
        J=J_value,
        a_1=1.0, 
        a_2=0.8,
        a_3=0.25,
        a_4=0.25,
        mu=0.22
    )

    # Add stochastic noise
    nsig = np.zeros((8, 76, 1))
    nsig[4] = model_pars['A'] * model_pars['a'] * (.320 - .120) * noise
    tvb_noise = tvb.noise.Additive(nsig=nsig, noise_seed=gv.random_seed)

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
        coupling=tvb.coupling.Linear(a=np.r_[0.1 * 10]),
        integrator=tvb.integrators.EulerStochastic(dt=0.1, noise=tvb_noise),
        stimulus=stimulus,
        monitors=(mon_tavg, mon_eeg),
        simulation_length=sim_time
    )
    sim.configure()

    (ttavg, tavg), (teeg, eeg) = sim.run(simulation_length=sim_time)

    return ttavg, tavg, teeg, eeg

def psd_simil_lorenzo(time_series_original, label, lowcut=0.5, highcut=45, n_seg=2048, window='hamming', return_psd=False, real_patient = False, patient = 0):
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

    psd_all = np.array(psd_all)
    psd_mean = np.mean(psd_all, axis=0)
    psd_std = np.std(psd_all, axis=0)

    # psd_all array dimensions are [n_regions, n_freqs]

    # Normalize
    normalization_factor = np.max(psd_mean)
    psd_mean_normalized = psd_mean.copy()
    psd_mean_normalized /= normalization_factor
    #psd_std /= normalization_factor

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, psd_mean_normalized, label=label, linewidth=3)
    #plt.fill_between(freqs, psd_mean + psd_std, psd_mean - psd_std, alpha=0.35)
    plt.title(label + ' mean PSD', fontsize=20)
    plt.ylim(0, 1.01)
    plt.xlim(0, 45)
    plt.xlabel('Frequency [Hz]', fontsize=16)
    plt.ylabel('PSD che dovrebbe essere giusta', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    if gv.show_plots:
        plt.show()
    else:
        plt.close()

def functional_connectivity_lorenzo(time_series_original, th=0.80):

    time_series = time_series_original.copy()
#     time_series = time_series[np.isnan(time_series)]

    for i in range(len(time_series[0,0,:,0])):
        time_series[:,0,i,0] = bandpass_filter(time_series[:,0,i,0], lowcut=0.5, highcut=45)
    tsr_corr = ev(time_series)

    corr= tsr_corr.array_data[..., 0, 0]
    
    corr -= np.eye(len(time_series[0,0,:,0]))

    abs_corr_mean = np.mean(np.abs(corr))
    corr_mean = np.mean(corr)

    #Define a threshold for connections
    th = th 

    corr_th = (corr) > th * np.max(corr)

    pmatrix = np.zeros((len(time_series[0,0,:,0]),len(time_series[0,0,:,0])))

    for i in range(len(time_series[0,0,:,0])):
        for j in range(len(time_series[0,0,:,0])):
            
            region_A = time_series[:,0, i, 0]
            region_B = time_series[:,0, j, 0]
            
            pmatrix[i][j] = pearsonr(region_A, region_B)[1]

    pmatrix_th = pmatrix < 0.05

    corr_th_significance = np.multiply(corr_th, pmatrix_th)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    # Plot corr_tavg
    im1 = ax1.imshow(corr, interpolation='nearest', cmap='bwr')
    #ax1.set_yticks(np.arange(len(conn.region_labels)))
    #ax1.set_yticklabels(conn.region_labels, fontsize=12)
    #ax1.set_xticks(np.arange(len(conn.region_labels)))
    #ax1.set_xticklabels(conn.region_labels, rotation='vertical', fontsize=12)
    ax1.set_title(f"FC lorenzo: {corr_mean:.4f} ({abs_corr_mean:.4f})", fontsize=29)

    # Plot corr_tavg_th
    im2 = ax2.imshow(corr_th, interpolation='nearest', cmap='bwr')
    #ax2.set_yticks(np.arange(len(conn.region_labels)))
    #ax2.set_yticklabels(conn.region_labels, fontsize=12)
    #ax2.set_xticks(np.arange(len(conn.region_labels)))
    #ax2.set_xticklabels(conn.region_labels, rotation='vertical', fontsize=12)
    ax2.set_title('FC lorenzo w/ threshold', fontsize=29)

    # Colorbar
    cax = fig.add_axes([ax2.get_position().x1 + 0.01, ax2.get_position().y0, 0.02, ax2.get_position().height])
    cbar = plt.colorbar(im2, cax=cax)

    plt.show()

def simulate_with_stimulus(sim_time=20000,
             tau_stage=gv.tau_stage,
             abeta_stage=gv.abeta_stage,
             g=gv.coup, #0.10004127233208432,
             velocity=gv.vel, #10.374658598387436,
             noise=gv.noi, #0.14480848489827142,
             a=gv.aa, #0.0728044836955774,
             b=gv.bb, #0.06293568322880852,
             factor=gv.factor, #0.04,#0.03999999999999998,#0.8356062043764902, #1.7591351809125855e-03, #0.8581280031180116,
             C_ip_values=np.linspace(33.75, 13.25, 10),
             cip=0,
             C_ep_values=np.linspace(108, 28, 10),
             cep=0,
             stimulus=None, 
             save_raw=False):
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
    conn.weights = conn.weights / 3 #np.max(conn.weights)
    conn.weights *= g
    conn.speed = np.array(velocity)
    conn.configure()

    # Model parameters (based on Jansen–Rit equations)
    Cep_value = C_ep_values[cep]
    Cip_value = C_ip_values[cip]
    J_value = 128
    J_stages = np.ones((76,1))

    tau_stage_1 = [ #2 areas
        24, 62  # PHC (parahippocampal gyrus)
    ]

    tau_stage_2 = [ #16 areas
        2, 40,   # AMYG (amygdala)
        9, 47,   # HC (hippocampus)
        13, 51,  # PCI (inferor parietal cortex)
        30, 68,  # TCC (central temporal)
        31, 69,  # TCI (inferior temporal)
        32, 70,  # TCPOL (temporal pole)
        33, 71,  # TCS (superior temporal)
        34, 72   # TCV (ventral temporal)
    ]

    tau_stage_3 = [ #34 areas
        0, 38,   # A1 (primary uditivity cortex)
        11, 49,  # IP (intraparietal cortex)
        12, 50,  # M1 (primary motor cortex)
        14, 52,  # PCIP (posterior inferor parietal cortex)
        15, 53,  # PCM (medial parietal cortex)
        16, 54,  # PCS (parietal sulcus)
        17, 55,  # PFCCL (caudolateral prefrontal cortex)
        18, 56,  # PFCDL (dorsolateral prefrontal cortex)
        19, 57,  # PFCDM (dorsomedial prefrontal cortex)
        21, 59,  # PFCORB (orbitofrontal)
        22, 60,  # PFCPOL (prefrontal polar)
        23, 61,  # PFCVL (ventrolateral prefrontal cortex)
        25, 63,  # PMCDL (ventrolateral premotor cortex)
        26, 64,  # PMCM (medial premotor cortex)
        27, 65,  # PMCVL (ventrolaterl premotor cortex)
        28, 66,  # S1 (primary somatosensorial cortex)
        35, 73,  # V1 (primary visul cortex)
    ]

    Abeta_stage_1 = [ #12 areas
        21, 59,   # PFCORB (orbitofrontal)
        22, 60,   # PFCPOL (prefrontal polar)
        30, 68,   # TCC (central temporal)
        31, 69,   # TCI (inferior temporal)
        32, 70,   # TCPOL (temporal pole)
        34, 72    # TCV (ventral temporal)
    ]

    Abeta_stage_2 = [ #22 areas
        2, 40,   # AMYG (amygdala)
        3, 41,   # CCA (anterior cingulate cortex)
        4, 42,   # CCP (posterior cingulate cortex)
        5, 43,   # CCR (rostral cingulate cortex)
        6, 44,   # CCS (subgenual cingulate cortex)
        9, 47,   # HC (hippocampus)
        10, 48,  # IA (insula = anterior + posterior)
        23, 61,  # PFCVL (ventrolateral prefrontal cortex)
        24, 62,  # PHC (parahippocampal gyrus)
        33, 71,  # TCS (superior temporal)
        37, 75   # CC (cingulate cortex)
    ]

    Abeta_stage_3 = [ #42 areas
        0, 38,   # A1 (primary uditivity cortex)
        1, 39,   # A2 (secondary uditivity cortex)
        7, 45,   # FEF (frontal eyes field)
        8, 46,   # G (gyrus)
        11, 49,  # IP (intraparietal cortex)
        12, 50,  # M1 (primary motor cortex)
        13, 51,  # PCI (inferor parietal cortex)
        14, 52,  # PCIP (posterior inferor parietal cortex)
        15, 53,  # PCM (medial parietal cortex)
        16, 54,  # PCS (parietal sulcus)
        17, 55,  # PFCCL (caudolateral prefrontal cortex)
        18, 56,  # PFCDL (dorsolateral prefrontal cortex)
        19, 57,  # PFCDM (dorsomedial prefrontal cortex)
        20, 58,  # PFCM (medial prefrontal cortex)
        25, 63,  # PMCDL (ventrolateral premotor cortex)
        26, 64,  # PMCM (medial premotor cortex)
        27, 65,  # PMCVL (ventrolaterl premotor cortex)
        28, 66,  # S1 (primary somatosensorial cortex)
        29, 67,  # S2 (secondary somatosensorial cortex)
        35, 73,  # V1 (primary visul cortex)
        36, 74,  # V2 (secondary visual cortex)
    ]

    Abeta_stage_1_second_trial = [  #alternative abeta staging
        30, 58, #Precuneus (PCu)
        35, 73, #Superior Frontal
        31, 69, #Superior Parietal
        28, 66, #Posterior Cingulate
        37, 75, #Middle Temporal
        36,	74  #Superior Temporal
    ]
        
    Abeta_stage_2_second_trial = [
        12, 50, #Hippocampus
        6, 44, #Entorhinal Cortex
        1, 39, #Amygdala
        13, 51, #Insula	
        32, 2, 70, 40, #Anterior Cingulate (Ros/Caud)
	    22, 60 #Parahippocampal Gyrus
    ]

    Abeta_stage_3_second_trial = [
        34, 72, #Thalamus
    	3, 41, #Caudate
	    33, 71, #Putamen
	    21, 59, #Pallidum
	    0, 38, #Accumbens
        ]

    degeneration = [0, 0, 0, (1/(1+19**(1/3))), (1/(1+19**(-1/3))), 0.95, 0.95, 0.95] 
    #stages derving from sigma function division in 3 parts (0, x1=1/3, x2=2/3, 1) s.t. f(0) = 0.05, f(1) = 0.95
    #alternative stagings, different between abeta and tau
    degeneration_tau = [0, 0, 0, 0.28, 0.78, 0.95, 0.95, 0.95]
    degeneration_abeta = [0, 0, 0, 0.6116, 0.8664, 0.95, 0.95, 0.95]
    degeneration_alpha = [[0, 0, 0], [0.11, 0.21, 0.31], [0.43, 0.53, 0.63], [0.75, 0.85, 0.95], [0.85, 0.95, 0.95], [0.95, 0.95, 0.95]]
    degeneration_alpha_2 = [[0, 0, 0], [0.21, 0.26, 0.31], [0.53, 0.58, 0.63], [0.85, 0.90, 0.95], [0.90, 0.95, 0.95], [0.95, 0.95, 0.95]]

    a_delta = (2.20*3.65/3.25)-1   ########################### Attenzione, valore di M
    b_delta = 13.25/33.75-1           #0.03133/0.0593-1

    a_array = np.ones((76,1))
    b_array = np.ones((76,1))*b #*(1+b_delta*0.01*tau_stage)
    regions_centers = conn.centres
    distance = np.ones((76,1))
    wide_y = np.max(regions_centers[:,1]) - np.min(regions_centers[:,1])
    for i in range(len(regions_centers[:,1])):
        region_coordinate_y = regions_centers[i,1]
        distance[i] = (region_coordinate_y - np.max(regions_centers[:,1]))/wide_y
        a_array[i] = a*(1 + distance[i]*factor)
    J_delta = 0.3

    #print('a iniziale', a_array[21:31])
    #print('b iniziale', b_array[21:31])
    #print('J iniziale', J_stages[21:31])
    for i in tau_stage_1:
        J_stages[i] = 1 - degeneration[tau_stage + 2]*J_delta
        b_array[i] = min(b_array[i], b*(1+b_delta*(1.8/2.2)*degeneration[tau_stage + 2]))
    
    for i in tau_stage_2:
        J_stages[i] = 1 - degeneration[tau_stage + 1]*J_delta
        b_array[i] = min(b_array[i], b*(1+b_delta*(1.8/2.2)*degeneration[tau_stage + 1]))

    for i in tau_stage_3:
        J_stages[i] = 1 - degeneration[tau_stage]*J_delta
        b_array[i] = min(b_array[i], b*(1+b_delta*(1.8/2.2)*degeneration[tau_stage]))

    for i in Abeta_stage_1:
        a_array[i] = a_array[i]*(1+a_delta*degeneration[abeta_stage + 2])
        b_array[i] = b_array[i] + b*b_delta*(0.4/2.2)*degeneration[abeta_stage+ 2]

    for i in Abeta_stage_2:
        a_array[i] = a_array[i]*(1+a_delta*degeneration[abeta_stage + 1])
        b_array[i] = b_array[i] + b*b_delta*(0.4/2.2)*degeneration[abeta_stage + 1]

    for i in Abeta_stage_3:
        a_array[i] = a_array[i]*(1+a_delta*degeneration[abeta_stage])
        b_array[i] = b_array[i] + b*b_delta*(0.4/2.2)*degeneration[abeta_stage]
    #print('a finale', a_array[21:31])
    #print('b finale', b_array[21:31])
    #print('J finale', J_stages[21:31])
    model_pars = dict(
        A=3.25, B=22, v0=6.0,
        a=a_array, b=b_array, r=0.56,
        nu_max=0.0025, J=J_value*J_stages,
        a_1=1.0, a_2 = Cep_value / J_value, #a_2 around 0.84375
        a_3=0.25, a_4 = Cip_value / J_value, #a_4 around 0.26367
        mu=0.22
    )

    # Add stochastic noise
    nsig = np.zeros((8, 76, 1))
    nsig[4] = model_pars['A'] * model_pars['a'] * (.320 - .120) * noise
    tvb_noise = tvb.noise.Additive(nsig=nsig, noise_seed=gv.random_seed)

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
        coupling=tvb.coupling.Linear(a=np.r_[a_array * 10]),
        integrator=tvb.integrators.EulerStochastic(dt=0.1, noise=tvb_noise),
        stimulus=stimulus,
        monitors=(mon_tavg, mon_eeg),
        simulation_length=sim_time
    )
    sim.configure()

    (ttavg, tavg), (teeg, eeg) = sim.run(simulation_length=sim_time)
    if save_raw==True:
        data_save_dir = os.path.join(gv.data_folder, f"tau{gv.tau_stage}_abeta{gv.abeta_stage}_rnd{gv.random_seed}")
        os.makedirs(data_save_dir, exist_ok=True)
        output = os.path.join(data_save_dir, f"eeg_and_teeg_raw_tau{gv.tau_stage}_abeta{gv.abeta_stage}_rnd{gv.random_seed}.npz")
        metadata = {
            "description": "File contains eeg (and time eeg) files and tavg (and time tavg) files"
        }

        np.savez_compressed(
            output,
            eeg=eeg,
            teeg=teeg,
            tavg=tavg,
            ttavg=ttavg,
            metadata=metadata
        )
            
    return ttavg, tavg, teeg, eeg

def new_parcellation_simulate(sim_time=20000,
             tau_stage=gv.tau_stage,
             abeta_stage=gv.abeta_stage,
             g=gv.coup, #0.10004127233208432,
             velocity=gv.vel, #10.374658598387436,
             noise=gv.noi, #0.14480848489827142,
             a=gv.aa, #0.0728044836955774,
             b=gv.bb, #0.06293568322880852,
             noise_seed=None,
             factor=gv.factor, #0.04,#0.03999999999999998,#0.8356062043764902, #1.7591351809125855e-03, #0.8581280031180116,
             C_ip_values=np.linspace(33.75, 13.25, 10),
             cip=0,
             C_ep_values=np.linspace(108, 28, 10),
             cep=0,
             stimulus=None, 
             freq= None,
             spot=None,
             intensity=None,
             save_raw=False,
             tms=False):
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
    conn = copy.deepcopy(get_conn())  # copia indipendente per ogni worker
    conn.weights = np.load(gv.struct_path_connectivity_86)[tau_stage, :, :]
    #mask = conn.weights < 100000
    #conn.weights[mask] = 0
    conn.weights = conn.weights / 820096.76119633  # .copy() non più necessario
    conn.weights = conn.weights * g
    conn.speed = np.array(velocity)
    conn.configure()

    region_number = 86

    if noise_seed is None:
        noise_seed = gv.random_seed
    if freq is None:
        freq = gv.high_tms_freq
    if spot is None:
        spot = gv.stimulation_spot
    if intensity is None:
        intensity = gv.tms_intensity

    # Model parameters (based on Jansen–Rit equations)
    Cep_value = C_ep_values[cep]
    Cip_value = C_ip_values[cip]
    J_value = 128
    J_stages = np.ones((region_number,1))

    a_array = np.ones((region_number,1))*a
    """
    print('a_array prima di essere calcolato')
    print(a_array.shape)
    print(a_array[:6])
    """
    b_array = np.ones((region_number,1))*b
    regions_centers = conn.centres[:84]
    distance = np.ones((region_number,1))
    wide_y = np.max(regions_centers[:,1]) - np.min(regions_centers[:,1])
    for i in range(len(regions_centers[:,1])):
        region_coordinate_y = regions_centers[i,1]
        distance[i] = (region_coordinate_y - np.max(regions_centers[:,1]))/wide_y
        a_array[i] = a*(1 + distance[i]*factor)

    """
    print('y massima', np.max(regions_centers[:,1]))
    print('y minima', np.min(regions_centers[:,1]))

    print('coordinate dei centri delle regioni')
    print(regions_centers.shape)
    print(regions_centers[:6])

    print('vettore delle distanze normalizzate')
    print(distance.shape)
    print(distance[:6])
    
    print('a_array dopo essere calcolato')
    print(a_array.shape)
    print(a_array[:6])
    """
    #Prova 1:
    """
    tau_stage_1 = [ #2 areas
        5,  54,   # EC   (entorhinal)
        15, 64    # PHIG (parahippocampal gyrus)
    ]

    tau_stage_2 = [ #16 areas
        7,  56,   # IPG (inferor parietal cortex)
        8,  57,   # ITG (inferior temporal)
        14, 63,   # MTG (middle temporal)
        29, 78,   # STG (superior temporal)
        32, 81,   # TP  (temporal pole)
        33, 82,   # TTG (trasverse temporal)
        40, 47,   # HI  (hippocampus)
        41, 48    # AM  (amygdala)
        ]

    tau_stage_3 = [ #34 areas
        0, 38,   # A1 (primary uditivity cortex)#######
        11, 49,  # IP (intraparietal cortex)########
        12, 50,  # M1 (primary motor cortex)##########
        22, 71,  # PCG (posterior cingulate)
        28, 77,  # SPG (superior parietal)
        7,  56,  # SPG (inferior parietal)
        16, 54,  # PCS (parietal sulcus)########
        2,  51,  # CACG (caudal anterior cingulate)
        3,  52,  # CMFG (caudal middle frontal)
        18, 56,  # PFCDL (dorsolateral prefrontal cortex)#######
        19, 57,  # PFCDM (dorsomedial prefrontal cortex)#######
        21, 59,  # PFCORB (orbitofrontal)########
        22, 60,  # PFCPOL (prefrontal polar)########
        23, 61,  # PFCVL (ventrolateral prefrontal cortex)#########
        25, 63,  # PMCDL (ventrolateral premotor cortex)#######
        26, 64,  # PMCM (medial premotor cortex)########
        27, 65,  # PMCVL (ventrolaterl premotor cortex)######
        28, 66,  # S1 (primary somatosensorial cortex)######
        35, 73,  # V1 (primary visual cortex)########
    ]
    
    #Prova 2:
    tau_stage_1 = [ # 4 regions
        5,  54,  # EC   (enthorinal cortex)
        15, 64   # PHIG (parahippocampal cortex)
    ]

    tau_stage_2 = [ # 30 regions
        2,  51,  # CACG (caudal anterior cingulate)
        7,  56,  # IPG  (inferior parietal cortex)
        8,  57,  # ITG  (inferior temporal gyrus)
        9,  58,  # ICG  (isthmus cingulate) 
        11, 60,  # LOFG (lateral orbitofrontal)
        13, 62,  # MOFG (medial orbitofrontal)
        14, 63,  # MTG  (middle temporal gyrus)
        24, 73,  # PCU  (precuneus)
        25, 74,  # RACG (rostral anterior cingulate)
        26, 75,  # RMFG (rostral middle frontal gyrus)
        27, 76,  # SFG  (superior frontal gyrus)
        28, 77,  # SPG  (superior parietal cortex)
        29, 78,  # STG  (superior temporal gyrus)
        40, 47,  # HI   (hippocampus)
        41, 48   # AM   (amygdala)
    ]

    tau_stage_3 = [ # 48 regions
        1,  50,  # BSTS (banks of superior temporal sulcus)
        3,  52,  # CMFG (caudal middle frontal gyrus)
        4,  53,  # CU   (cuneus)
        6,  55,  # FG   (fusiform gyrus)
        10, 59,  # LOG  (lateral occipital)
        12, 61,  # LG   (lingual gyrus)
        16, 65,  # PaCG (paracentral lobule)
        17, 66,  # POP  (pars opercularis)
        18, 67,  # POR  (pars orbitalis)
        19, 68,  # PTR  (pars triangularis)
        20, 69,  # PCAL (pericalcarine / primary visual cortex)
        21, 70,  # PoCG (postcentral gyrus / primary somatosensory cortex)
        23, 72,  # PrCG (precentral gyrus / primary motor cortex)
        30, 79,  # SMG  (supramarginal gyrus)
        31, 80,  # FP   (frontal temporal)
        32, 81,  # TP   (temporal pole)
        33, 82,  # TTG  (transverse temporal gyrus / primary auditory cortex)
        34, 83,  # IN   (insula)
        35, 84,  # CER  (cerebellum cortex)
        36, 43,  # TH   (thalamus)
        37, 44,  # CA   (caudate)
        38, 45,  # PU   (putamen)
        39, 46,  # PA   (pallidum)
        42, 49,  # AC   (accumbens) 
    ]

    #Prova 1:
    
    Abeta_stage_1 = [ #14 areas
        
        8,  57,   # ITG (inferior temporal)
        11, 60,   # LOFG (lateral orbitofrontal)
        13, 62,   # MOFG (medial orbitofrontal)
        14, 63,   # MTG (middle temporal)
        18, 67,   # POR (parsorbitalis)
        32, 81,   # TP  (temporal pole)
        33, 82,   # TTG (trasverse temporal)
    ]

    Abeta_stage_2 = [ #18 areas
        2,  51,  # CACG (caudal anterior cingulate)
        9,  58,  # ICG (isthmus cingulate)
        15, 64,  # PHIG (parahippocampal)
        22, 71,  # PCG (posterior cingulate)
        25, 74,  # RACG (rostral anterior cingulate)
        29, 78,  # STG (superior temporal)
        34, 83,  # IN (insula)
        40, 47,  # HI (hippocampus)
        41, 48   # AM (amygdala)
        #23, 61  # PFCVL (ventrolateral prefrontal cortex)#########
    ]

    Abeta_stage_3 = [ #52 areas
        1,  50,  # BSTS  (bankssts)
        3,  52,  # CMFG  (caudalmiddlefrontal)
        4,  53,  # CU    (cuneus)
        5,  54,  # EC    (entorhinal)
        6,  55,  # FG    (fusiform)
        7,  56,  # IPG   (inferiorparietal)
        10, 59,  # LOG   (lateraloccipital)
        12, 61,  # LG    (lingual)
        16, 65,  # PaCG  (paracentral)
        17, 66,  # POP   (parsopercularis)
        19, 68,  # PTR   (parstriangularis)
        20, 69,  # PCAL  (pericalcarine)
        21, 70,  # PoCG  (postcentral)
        23, 72,  # PrCG  (precentral)
        24, 73,  # PCU   (precuneus)
        26, 75,  # RMFG  (rostralmiddlefrontal)
        27, 76,  # SFG   (superiorfrontal)
        28, 77,  # SPG   (superiorparietal)
        30, 79,  # SMG   (supramarginal)
        31, 80,  # FP    (frontalpole)
        35, 84,  # CER   (Cerebellum-Cortex)
        36, 43,  # TH    (Thalamus)
        37, 44,  # CA    (Caudate)
        38, 45,  # PU    (Putamen)
        39, 46,  # PA    (Pallidum)
        42, 49   # AC    (Accumbens-area)
    ]
    

    Abeta_stage_1 = [ # 26 regions
        1,  50,
        3,  52,
        6,  55,
        8,  57,
        11, 60,
        13, 62,
        14, 63,
        18, 67,
        26, 75,
        27, 76,
        29, 78,        
        31, 80, 
        32, 81
    ]

    Abeta_stage_2 = [ # 22 regions
        2,  51,
        5,  54,
        9,  58,
        15, 64,
        17, 66,
        19, 68,
        22, 71,
        25, 74,
        34, 83,
        40, 47,
        41, 48
    ]

    Abeta_stage_3 = [ # 36 regions
        4,  53,
        7,  56,
        10, 59,
        12, 61,
        16, 65,
        20, 69,
        21, 70,
        23, 72,
        24, 73,
        28, 77,
        30, 79,
        33, 82,
        35, 84,
        36, 43,
        37, 44,
        38, 45,
        39, 46,
        42, 49
    ]
    """
    """
    Abeta_stage_1 = [ # 9 regions
        3, 11, 19, 31, 34, 58, 68, 80, 83
    ]

    Abeta_stage_2 = [ # 34 regions
        1, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 26, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 47,
        48, 50, 53, 54, 56, 57, 59, 61, 63, 84, 85
    ]

    Abeta_stage_3 = [ # 43 regions
        0, 2, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 32, 33, 44, 46, 49, 51,
        52, 55, 60, 62, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 81, 82
    ]

    tau_stage_1 = [ # 4 regions
        4, 40, 47, 53
    ]

    tau_stage_2 = [ # 17 regions
        7, 13, 14, 16, 18, 31, 37, 38, 39, 41, 45, 65, 66, 67, 80, 84, 85
    ]

    tau_stage_3 = [ # 38 regions
        0, 2, 3, 6, 9, 11, 15, 17, 19, 21, 22, 23, 24, 25, 27, 29, 33, 34, 44, 46, 49, 51, 52,
        55, 58, 62, 64, 68, 70, 71, 72, 73, 74, 75, 76, 81, 82, 83
    ]
    """
    Abeta_stage_1 = [ # 8 regions
        0, 5, 7, 31, 34, 54, 56, 83
    ]

    Abeta_stage_2 = [ # 26 regions
        4, 8, 13, 14, 21, 24, 35, 36, 38, 39, 40, 41, 42, 43, 45,
        46, 47, 48, 57, 60, 63, 70, 73, 80, 84, 85
    ]

    Abeta_stage_3 = [ # 52 regions
        1, 2, 3, 6, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 22, 23,
        25, 26, 27, 28, 29, 30, 32, 33, 37, 44, 49, 50, 51, 52, 53,
        55, 58, 59, 61, 62, 64, 65, 66, 67, 68, 69, 71, 72, 74, 75,
        76, 77, 78, 79, 81, 82
    ]

    tau_stage_1 = [ # 2 regions
    14, 39
    ]

    tau_stage_2 = [ # 16 regions
        0, 4, 5, 7, 13, 20, 29, 34, 35, 40, 46, 54, 56, 63, 78, 83
    ]

    tau_stage_3 = [ # 38 regions
        1, 6, 11, 12, 15, 16, 17, 18, 19, 22, 23, 25, 26, 27, 28, 30,
        31, 33, 37, 49, 50, 51, 52, 53, 55, 58, 59, 61, 64, 66, 68,
        69, 71, 72, 75, 76, 79, 81
    ]

    degeneration = [0, 0, 0, (1/(1+19**(1/3))), (1/(1+19**(-1/3))), 0.95, 0.95, 0.95] 

    a_delta = (2.20*3.65/3.25)-1   ########################### Attenzione, valore di M
    b_delta = 13.25/33.75-1           #0.03133/0.0593-1
    J_delta = 0.3
    """
    print('a iniziale', a_array[0:8])
    print('b iniziale', b_array[0:8])
    print('J iniziale', J_stages[0:8])
    """
    for i in tau_stage_1:
        J_stages[i] = 1 - degeneration[tau_stage + 2]*J_delta
        b_array[i] = min(b_array[i], b*(1+b_delta*(1.8/2.2)*degeneration[tau_stage + 2]))
    
    for i in tau_stage_2:
        J_stages[i] = 1 - degeneration[tau_stage + 1]*J_delta
        b_array[i] = min(b_array[i], b*(1+b_delta*(1.8/2.2)*degeneration[tau_stage + 1]))

    for i in tau_stage_3:
        J_stages[i] = 1 - degeneration[tau_stage]*J_delta
        b_array[i] = min(b_array[i], b*(1+b_delta*(1.8/2.2)*degeneration[tau_stage]))

    for i in Abeta_stage_1:
        a_array[i] = a_array[i]*(1+a_delta*degeneration[abeta_stage + 2])
        b_array[i] = b_array[i] + b*b_delta*(0.4/2.2)*degeneration[abeta_stage+ 2]

    for i in Abeta_stage_2:
        a_array[i] = a_array[i]*(1+a_delta*degeneration[abeta_stage + 1])
        b_array[i] = b_array[i] + b*b_delta*(0.4/2.2)*degeneration[abeta_stage + 1]

    for i in Abeta_stage_3:
        a_array[i] = a_array[i]*(1+a_delta*degeneration[abeta_stage])
        b_array[i] = b_array[i] + b*b_delta*(0.4/2.2)*degeneration[abeta_stage]
    """
    print('a finale', a_array[0:8])
    print('b finale', b_array[0:8])
    print('J finale', J_stages[0:8])
    """

    mltlow = [-1,-1, 1, 1,-1]#[-0.2,-0.2, 0.6, 0.6]#A, a, B, b  # -0.2 # max_long_term_depotentiation_percentage proviamo a mettere -1 e +1 ai pesi
    mlthigh =[0 ,-1,-1,-1, 1]#[0   ,-0.2,-0.2,-0.2]#A, a, B, b # 0.6 # max_long_term_potentiation_percentage

    Cep_value_array = Cep_value*np.ones((region_number,1))
    Cip_value_array = Cip_value*np.ones((region_number,1))
    A_value = 3.25
    B_value = 22
    
    A_value_array = A_value*np.ones((region_number,1))
    B_value_array = B_value*np.ones((region_number,1))
    subth_param = np.array([0.4, 0.6])
    suprath_param = np.array([1.0, 0.6])

    print('Parameter of the actual simulation')
    print('stimulation_spot', spot)
    print('high_tms_freq', freq)
    print('abeta', abeta_stage)
    print('tau', tau_stage)
    print('Noise seed utlizzato:',noise_seed)
    print('intensity:',intensity)

    if freq == False:
        mlt = mltlow
        searching_freq = 0.5
        print('depression')
    else:
        mlt = mlthigh
        searching_freq = 10
        print('potentiation')

    """
    if gv.supra_threshold == True:
        par = suprath_param
    else:
        par = subth_param

    if gv.potent_effect == True:
        mlt = mltpot
    else:
        mlt = mltdep
    """
    """
    #print('tms_param_variation', gv.tms_param_variation)
    if gv.tms_param_variation == True:
        
        #file_name = f"super_th_reg_a{abeta_stage}_t{tau_stage}_f{gv.frequence}_s{gv.stimulation_spot}"
        #df = pd.read_pickle(os.path.join('../Tms_app_results_2', file_name))
        #suprath_regions = df["is_greater"]
        #stim_amplitudes = df["mean_amp_post"]
                
        for i in range(region_number):
            #print('HO CAMBIATO I PARAMETRI')
            if gv.stimulations_amplitude_0_0[i]/gv.baseline_amplitude_0_0[i] > 1.5:#suprath_regions[i]==True: #
                par = suprath_param
            else:
                par = subth_param
            
            par = suprath_param
            #Cep_value_array[i] += Cep_value_array[i]*mlt*par[0]*gv.stimulations_amplitude_3_3[i]/np.max(gv.stimulations_amplitude_0_0)
            #Cip_value_array[i] += Cip_value_array[i]*mlt*par[1]*gv.stimulations_amplitude_3_3[i]/np.max(gv.stimulations_amplitude_0_0)#
            A_value_array[i] += A_value_array[i]*mlt*par[0]*gv.stimulations_amplitude_3_3[i]/np.max(gv.stimulations_amplitude_0_0)#stim_amplitudes[i]/np.max(stim_amplitudes)#
            B_value_array[i] += B_value_array[i]*mlt*par[1]*gv.stimulations_amplitude_3_3[i]/np.max(gv.stimulations_amplitude_0_0)#stim_amplitudes[i]/np.max(stim_amplitudes)#
            #J_stages[i] += J_stages[i]*tms_percentage_effects*gv.stimulations_amplitude_3_3[i]/np.max(gv.stimulations_amplitude_0_0)

        old_weights = conn.weights
        new_weights = conn.weights
        for i in range(region_number):
            
            if gv.stimulations_amplitude_0_0[i]/gv.baseline_amplitude_0_0[i] > 1.5:#suprath_regions[i]==True: #
                par = suprath_param
            else:
                par = subth_param
            
            par=suprath_param
            new_weights[i,:] = new_weights[i,:] + 0.5*old_weights[i,:]*mlt*par[0]*gv.stimulations_amplitude_3_3[i]/np.max(gv.stimulations_amplitude_0_0)#stim_amplitudes[i]/np.max(stim_amplitudes)#
            new_weights[:,i] = new_weights[:,i] + 0.5*old_weights[:,i]*mlt*par[0]*gv.stimulations_amplitude_3_3[i]/np.max(gv.stimulations_amplitude_0_0)#stim_amplitudes[i]/np.max(stim_amplitudes)#

        conn.weights = new_weights
    """

    #norm = 0.75
    norm = 0.2 #prima era 0.2
    #print('A_array originale')
    #print(A_value_array)
    if gv.tms_param_variation == True:
        print("STO CAMBIANDO I PARAMETRI")

        file_name = f"super_th_reg_a{abeta_stage}_t{tau_stage}_f{searching_freq}_s{spot}"
        base_dir = os.path.dirname(os.path.abspath(__file__))  # cartella di utils.py
        path = os.path.join(base_dir, '..', 'Tms_results_end_18', file_name)
        df = pd.read_pickle(path)

        intensityplus = intensity + 0.01
        intensityminus = intensity - 0.01
        df=df.query("sf <= @intensityplus and sf >= @intensityminus")
        
        pre_stim_amp = df["mean_amp_pre"].reset_index(drop=True)
        post_stim_amp = df["mean_amp_post"].reset_index(drop=True)
        baseline = df["mean_baseline"].reset_index(drop=True)
        threshold = df["is_greater"].reset_index(drop=True)

        #print(post_stim_amp)
                
        for i in range(region_number):
            #print('HO CAMBIATO I PARAMETRI')
            if  threshold[i] == True:
                par = suprath_param
            else:
                par = subth_param
                #print('sub region ', i)

            #par = subth_param

            #Cep_value_array[i] += Cep_value_array[i]*mlt*par[0]*gv.stimulations_amplitude_3_3[i]/np.max(gv.stimulations_amplitude_0_0)
            #Cip_value_array[i] += Cip_value_array[i]*mlt*par[1]*gv.stimulations_amplitude_3_3[i]/np.max(gv.stimulations_amplitude_0_0)#
            A_value_array[i] += A_value_array[i]*mlt[0]*par[0]*(np.abs(post_stim_amp[i]-pre_stim_amp[i]))/norm#(np.abs(post_stim_amp[i]))/norm#/np.abs(baseline[i])
            B_value_array[i] += B_value_array[i]*mlt[2]*par[1]*(np.abs(post_stim_amp[i]-pre_stim_amp[i]))/norm#(np.abs(post_stim_amp[i]))/norm#/np.abs(baseline[i])
            a_array[i]       +=       a_array[i]*mlt[1]*par[0]*(np.abs(post_stim_amp[i]-pre_stim_amp[i]))/norm#(np.abs(post_stim_amp[i]))/norm#/np.abs(baseline[i])
            b_array[i]       +=       b_array[i]*mlt[3]*par[1]*(np.abs(post_stim_amp[i]-pre_stim_amp[i]))/norm#(np.abs(post_stim_amp[i]))/norm#/np.abs(baseline[i])
            #J_stages[i] += J_stages[i]*tms_percentage_effects*gv.stimulations_amplitude_3_3[i]/np.max(gv.stimulations_amplitude_0_0)

        old_weights = conn.weights.copy()
        new_weights = conn.weights.copy()
        for i in range(region_number):
            
            if threshold[i] == True:
                par = suprath_param
            else:
                par = subth_param
            
            #par = subth_param
            # prima moltiplicavo i pesi per mlt[2] e non per mlt[4]
            new_weights[i,:] = new_weights[i,:] + 0.5*old_weights[i,:]*mlt[4]*par[0]*(np.abs(post_stim_amp[i]-pre_stim_amp[i]))/norm#(np.abs(post_stim_amp[i]))/norm#/np.abs(baseline[i])
            new_weights[:,i] = new_weights[:,i] + 0.5*old_weights[:,i]*mlt[4]*par[0]*(np.abs(post_stim_amp[i]-pre_stim_amp[i]))/norm#(np.abs(post_stim_amp[i]))/norm#/np.abs(baseline[i])

        conn.weights = new_weights
    
    #print('A_array curato')
    #print(A_value_array)
    model_pars = dict(
        A=A_value_array, B=B_value_array, v0=6.0,
        a=a_array, b=b_array, r=0.56,
        nu_max=0.0025, J=J_value*J_stages,
        a_1=1.0, a_2 = Cep_value_array / J_value, #a_2 around 0.84375
        a_3=0.25, a_4 = Cip_value_array / J_value, #a_4 around 0.26367
        mu=0.22
    )

    # Add stochastic noise
    nsig = np.zeros((8, region_number, 1))
    nsig[4] = model_pars['A'] * model_pars['a'] * (.320 - .120) * noise
    tvb_noise = tvb.noise.Additive(nsig=nsig, noise_seed=noise_seed)

    # Define monitors
    mon_tavg = monitors.TemporalAverage(period=gv.temp_avg_period)
    mon_EEG = get_eeg_monitor(conn=conn)

    mons = (
        mon_tavg,                              # results[0] -> (ttavg, tavg)
        mon_EEG,                               # results[1] -> (teeg, eeg)
        monitors.ProgressLogger(period=10000)  # results[2]
    )
    mons[0].configure()
    mons[1].configure()
    mons[2].configure()
    
    sim = Simulator_Modified(        
        model=JRPSP(
                variables_of_interest=("y0", "y1", "y2", "y3", "y4", "y5", "y6", "y7"),
                **{k: np.array(v) for k, v in model_pars.items()}
            ),        
        connectivity=conn,      
        coupling=tvb.coupling.Linear(a=np.r_[a_array * 10]),     
        conduction_speed=float(conn.speed),       
        integrator=tvb.integrators.EulerStochastic(dt=0.1, noise=tvb_noise),     
        monitors=mons,        
        surface=None,        
        stimulus=stimulus,
        simulation_length=sim_time
    ) 
    sim.configure()
        #print(sim.model.__dict__)
    results = sim.run(simulation_length=sim_time)

    ttavg, tavg = results[0]   # TemporalAverage
    teeg, eeg   = results[1]   # EEG
 
    return ttavg, tavg, teeg, eeg

def deterministic_new_parcellation_simulate(sim_time=20000,
             tau_stage=gv.tau_stage,
             abeta_stage=gv.abeta_stage,
             g=gv.coup, #0.10004127233208432,
             velocity=gv.vel, #10.374658598387436,
             noise=gv.noi, #0.14480848489827142,
             a=gv.aa, #0.0728044836955774,
             b=gv.bb, #0.06293568322880852,
             factor=gv.factor, #0.04,#0.03999999999999998,#0.8356062043764902, #1.7591351809125855e-03, #0.8581280031180116,
             C_ip_values=np.linspace(33.75, 13.25, 10),
             cip=0,
             C_ep_values=np.linspace(108, 28, 10),
             cep=0,
             stimulus=None, 
             save_raw=False):
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
    conn = copy.deepcopy(get_conn())  # copia indipendente per ogni worker
    conn.weights = np.load(gv.struct_path_connectivity_86)[tau_stage, :, :]
    conn.weights = conn.weights / 820096.76119633  # .copy() non più necessario
    conn.weights = conn.weights * g
    conn.speed = np.array(velocity)
    conn.configure()

    region_number = 86
    # Model parameters (based on Jansen–Rit equations)
    Cep_value = C_ep_values[cep]
    Cip_value = C_ip_values[cip]
    J_value = 128
    J_stages = np.ones((region_number,1))

    a_array = np.ones((region_number,1))*a
    b_array = np.ones((region_number,1))*b

    Abeta_stage_1 = [ # 8 regions
        0, 5, 7, 31, 34, 54, 56, 83
    ]

    Abeta_stage_2 = [ # 26 regions
        4, 8, 13, 14, 21, 24, 35, 36, 38, 39, 40, 41, 42, 43, 45,
        46, 47, 48, 57, 60, 63, 70, 73, 80, 84, 85
    ]

    Abeta_stage_3 = [ # 52 regions
        1, 2, 3, 6, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 22, 23,
        25, 26, 27, 28, 29, 30, 32, 33, 37, 44, 49, 50, 51, 52, 53,
        55, 58, 59, 61, 62, 64, 65, 66, 67, 68, 69, 71, 72, 74, 75,
        76, 77, 78, 79, 81, 82
    ]

    tau_stage_1 = [ # 2 regions
    14, 39
    ]

    tau_stage_2 = [ # 16 regions
        0, 4, 5, 7, 13, 20, 29, 34, 35, 40, 46, 54, 56, 63, 78, 83
    ]

    tau_stage_3 = [ # 38 regions
        1, 6, 11, 12, 15, 16, 17, 18, 19, 22, 23, 25, 26, 27, 28, 30,
        31, 33, 37, 49, 50, 51, 52, 53, 55, 58, 59, 61, 64, 66, 68,
        69, 71, 72, 75, 76, 79, 81
    ]

    degeneration = [0, 0, 0, (1/(1+19**(1/3))), (1/(1+19**(-1/3))), 0.95, 0.95, 0.95] 

    a_delta = (2.20*3.65/3.25)-1   ########################### Attenzione, valore di M
    b_delta = 13.25/33.75-1           #0.03133/0.0593-1
    J_delta = 0.3
    for i in tau_stage_1:
        J_stages[i] = 1 - degeneration[tau_stage + 2]*J_delta
        b_array[i] = min(b_array[i], b*(1+b_delta*(1.8/2.2)*degeneration[tau_stage + 2]))
    
    for i in tau_stage_2:
        J_stages[i] = 1 - degeneration[tau_stage + 1]*J_delta
        b_array[i] = min(b_array[i], b*(1+b_delta*(1.8/2.2)*degeneration[tau_stage + 1]))

    for i in tau_stage_3:
        J_stages[i] = 1 - degeneration[tau_stage]*J_delta
        b_array[i] = min(b_array[i], b*(1+b_delta*(1.8/2.2)*degeneration[tau_stage]))

    for i in Abeta_stage_1:
        a_array[i] = a_array[i]*(1+a_delta*degeneration[abeta_stage + 2])
        b_array[i] = b_array[i] + b*b_delta*(0.4/2.2)*degeneration[abeta_stage+ 2]

    for i in Abeta_stage_2:
        a_array[i] = a_array[i]*(1+a_delta*degeneration[abeta_stage + 1])
        b_array[i] = b_array[i] + b*b_delta*(0.4/2.2)*degeneration[abeta_stage + 1]

    for i in Abeta_stage_3:
        a_array[i] = a_array[i]*(1+a_delta*degeneration[abeta_stage])
        b_array[i] = b_array[i] + b*b_delta*(0.4/2.2)*degeneration[abeta_stage]

    model_pars = dict(
        A=3.25, B=22, v0=6.0,
        a=a_array, b=b_array, r=0.56,
        nu_max=0.0025, J=J_value*J_stages,
        a_1=1.0, a_2 = Cep_value / J_value, #a_2 around 0.84375
        a_3=0.25, a_4 = Cip_value / J_value, #a_4 around 0.26367
        mu=0
    )

    # Add stochastic noise
    nsig = np.zeros((8, region_number, 1))

    # Define monitors
    mon_tavg = monitors.TemporalAverage(period=gv.temp_avg_period)
    mon_EEG = get_eeg_monitor()

    mons = (
        mon_tavg,                              # results[0] -> (ttavg, tavg)
        mon_EEG,                               # results[1] -> (teeg, eeg)
        monitors.ProgressLogger(period=10000)  # results[2]
    )
    mons[0].configure()
    mons[1].configure()
    mons[2].configure()

    initial_conditions = np.zeros((1, 8, region_number, 1))
    initial_conditions[:,0,:,:] = 0.00363
    initial_conditions[:,1,:,:] = 1.06187
    initial_conditions[:,2,:,:] = 2.42773
    initial_conditions[:,6,:,:] = 0.00121
    
    sim = Simulator_Modified(        
        model=JRPSP(
                variables_of_interest=("y0", "y1", "y2", "y3", "y4", "y5", "y6", "y7"),
                **{k: np.array(v) for k, v in model_pars.items()}
            ),        
        connectivity=conn,      
        coupling=tvb.coupling.Linear(a=np.r_[a_array * 10]),     
        conduction_speed=float(conn.speed),
        integrator = tvb.integrators.HeunDeterministic(dt=0.1),
        monitors=mons,        
        surface=None,        
        stimulus=stimulus,
        initial_conditions=initial_conditions,  
        simulation_length=sim_time
    ) 
    sim.configure()
        #print(sim.model.__dict__)
    results = sim.run(simulation_length=sim_time)

    ttavg, tavg = results[0]   # TemporalAverage
    teeg, eeg   = results[1]   # EEG
 
    return ttavg, tavg, teeg, eeg

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
        stimulus_onset = 0
        mask1 = 0
        mask2 = 4000
        for key, value in datas_stim.items():
            tavg_stim = value['tavg']
            eeg_stim = value['eeg']
            label = value['label']
            color = value['color']
            mean_n1 = mean_channels(stimulus_onset+mask1, stimulus_onset+mask2, eeg_stim, chlist)
            std_n1 =  std_channels(stimulus_onset+mask1, stimulus_onset+mask2, eeg_stim, chlist)
            plt.plot(teeg_stim[stimulus_onset+mask1:stimulus_onset+mask2], mean_n1, color = color, linewidth = 3, alpha = 0.55)
            plt.fill_between(teeg_stim[stimulus_onset+mask1:stimulus_onset+mask2], mean_n1 + std_n1, mean_n1 - std_n1, color = color, linewidth = 3, alpha = 0.35 )
            windowed_signal = mean_n1  # mean_n1 è già sliced con mask1:mask2
            #print(label, 'N1 depth is:', np.min(windowed_signal))
            #print(label, 'N1 peak is:', np.max(windowed_signal))
            #print(label, 'N1 peak-to-peak:', np.max(windowed_signal) - np.min(windowed_signal))

        plt.axvline(x=2500, color='k', linestyle='--', linewidth=1.5, label='Stimulus onset')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.title("EEG sensors Cz", fontsize = 29)
        plt.xlabel('Time (ms)', fontsize = 29)
        plt.ylabel(r'Voltage (10 $\mu$V)', fontsize = 29)
        plt.xticks(np.linspace(stimulus_onset+mask1, stimulus_onset+mask2, 21))#[mask1, mask2], [mask1-stimulus_onset,mask2-stimulus_onset], fontsize = 22) #1500 is stimulus onset 
        plt.yticks(np.linspace(0,1,21))
        plt.xlim(stimulus_onset+mask1+1000, stimulus_onset+mask2)
        #plt.yticks(fontsize = 22)
        plt.legend(fontsize = 29)
        plt.grid()
        if gv.save_plots == True:
            plt.savefig(r"C:\Users\User\OneDrive - University of Pisa\Desktop\Immagini da far vedere a Lorenzo_20_03\Stimulation_channels_Pz_CPz\20_mean.png")
        if gv.show_plots == True:
            plt.show()
        plt.close()

def compute_new_parameters(no_stim_ratio, stim_ratio, maximum_value, parameter):
    parameter_new = (stim_ratio/no_stim_ratio)*maximum_value*parameter
    return parameter_new
    

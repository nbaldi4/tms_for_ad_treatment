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
import tvb.simulator.lab as tvb
from tvb.simulator.lab import equations,patterns, monitors
from tvb.simulator.lab import *
from tvb.datatypes import graph
from tvb.datatypes.time_series import TimeSeriesRegion 
from scipy.stats import pearsonr, wasserstein_distance
from scipy.signal import butter, filtfilt
from scipy import signal

from classes import JRPSP
from global_variable_creation import get_eeg_monitor, get_conn
import global_variable_creation as gv


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

def ev(time_series):
    conn = get_conn()
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

def functional_connectivity(time_series, label):

    conn = get_conn()
    #time_series = time_series[np.isnan(time_series)]
    tsr_corr = ev(time_series)

    corr= tsr_corr.array_data[..., 0, 0]
    
    corr -= np.eye(len(time_series[0,0,:,0]))

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    # Plot corr_tavg
    im1 = ax1.imshow(corr, interpolation='nearest', cmap='bwr')
    ax1.set_yticks(np.arange(len(conn.region_labels)))
    ax1.set_yticklabels(conn.region_labels, fontsize=12)
    ax1.set_xticks(np.arange(len(conn.region_labels)))
    ax1.set_xticklabels(conn.region_labels, rotation='vertical', fontsize=12)
    ax1.set_title(label+' FC', fontsize=29)

    # Plot corr_tavg_th
    im2 = ax2.imshow(corr_th, interpolation='nearest', cmap='bwr')
    ax2.set_yticks(np.arange(len(conn.region_labels)))
    ax2.set_yticklabels(conn.region_labels, fontsize=12)
    ax2.set_xticks(np.arange(len(conn.region_labels)))
    ax2.set_xticklabels(conn.region_labels, rotation='vertical', fontsize=12)
    ax2.set_title(label+' FC w/ threshold', fontsize=29)

    # Colorbar
    cax = fig.add_axes([ax2.get_position().x1 + 0.01, ax2.get_position().y0, 0.02, ax2.get_position().height])
    cbar = plt.colorbar(im2, cax=cax)

    if gv.save_plots:
        fig_name = f"Functional_connectivity_{label}.png"
        os.makedirs(gv.fig_folder, exist_ok=True)
        plt.savefig(os.path.join(gv.fig_folder, fig_name), dpi=200)
    if gv.show_plots:
        plt.show()

def functional_connectivity_alpha(time_series, label):
    """
    Compute the alpha-band (8–13 Hz) functional connectivity matrix.

    This is a convenience wrapper around `functional_connectivity()`.

    Args:
        time_series (np.ndarray): Regional time series array.
        label (str): Dataset label.

    Returns:
        np.ndarray: Alpha-band functional connectivity matrix.
    """

    time_series = bandpass_filter(time_series, lowcut=8, highcut=13)
    return functional_connectivity(time_series, label)

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

def simulate(sim_time=20000, cp=0, np_parameter=3,
             g=1,
             velocity=10,
             noise=0.05,
             a=0.1,
             b=0.05,
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
    conn.weights = np.load(gv.struct_path)[cp, np_parameter, :, :]
    conn.weights = conn.weights / np.max(conn.weights)
    conn.weights *= g
    conn.speed = np.array(velocity)
    conn.configure()

    # Model parameters (based on Jansen–Rit equations)
    Cep_value = C_ep_values[cep]
    Cip_value = C_ip_values[cip]
    J_value = 128

    model_pars = dict(
        A=3.25, B=22, v0=6.0,
        a=a, b=b, r=0.56,
        nu_max=0.0025, J=J_value,
        a_1=1.0, a_2=Cep_value / J_value,
        a_3=0.25, a_4=Cip_value / J_value,
        mu=0.22
    )

    # Add stochastic noise
    nsig = np.zeros((8, 76, 1))
    nsig[4] = model_pars['A'] * model_pars['a'] * (.320 - .120) * noise
    noise = tvb.noise.Additive(nsig=nsig)

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
        integrator=tvb.integrators.EulerStochastic(dt=0.1, noise=noise),
        stimulus=stimulus,
        monitors=(mon_tavg, mon_eeg),
        simulation_length=sim_time
    )
    sim.configure()

    (ttavg, tavg), (teeg, eeg) = sim.run(simulation_length=sim_time)
    return ttavg, tavg, teeg, eeg

def psd(time_series, label, lowcut=0.5, highcut=45, n_seg=2048, window='hamming', return_psd=False):
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

    # Normalize
    psd_mean /= np.max(psd_mean)
    psd_std /= np.max(psd_mean)

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
        fig_name = f"{label}_PSD.png"
        os.makedirs(gv.fig_folder, exist_ok=True)
        plt.savefig(os.path.join(gv.fig_folder, fig_name), dpi=200)
    if gv.show_plots:
        plt.show()
    else:
        plt.close()
    if return_psd:
        return freqs, psd_mean

"""
def NSGA_compare_matrix_and_computed_psd(path, time_series, n_bin=180, fs=100):
    # PSD reale
    matrix = np.load(path)
    real_psd_mean = np.mean(matrix, axis=0)  # media su tutti i campioni

    # PSD simulata
    simulated_psd_mean = NSGA_psd(time_series=time_series, n_bin=n_bin, fs=fs)

    # Assicurati che abbiano la stessa lunghezza
    min_len = min(len(real_psd_mean), len(simulated_psd_mean))
    real_psd_mean = real_psd_mean[:min_len]
    simulated_psd_mean = simulated_psd_mean[:min_len]

    # Differenze assolute per bin
    diff_per_bin = np.abs(real_psd_mean - simulated_psd_mean)

    # Somma totale delle differenze
    total_diff = np.sum(diff_per_bin)

    return total_diff
"""
    
def real_vs_simulated_PSD(matrix, time_series, n_bin=180, fs=100,
                                   label_real="Experimental", label_sim="Simulated", return_value = False):

    # =========================
    # PSD reale
    # =========================
    real_psd_mean = np.mean(matrix, axis=0)

    # Taglio e normalizzazione reale
    real_psd_mean = real_psd_mean[:n_bin].astype(float)

    # Evito divisione per zero
    if np.max(real_psd_mean) == 0:
        real_psd_mean = real_psd_mean + 1e-12
    real_psd_mean /= np.max(real_psd_mean)


    # Frequenze reali (uniformi)
    freqs_uniform = np.linspace(0, 45, n_bin)

    # =========================
    # PSD simulata
    # =========================
    simulated_freqs, simulated_psd_mean = psd(time_series=time_series, label='sim_psd', return_psd=True)

    # Taglio frequenze a 45 Hz
    mask = simulated_freqs <= 45
    simulated_freqs = simulated_freqs[mask]
    simulated_psd_mean = simulated_psd_mean[mask]

    # =========================
    # Interpolazione simulata su 180 bin
    # =========================
    simulated_psd_resampled = np.interp(freqs_uniform, simulated_freqs, simulated_psd_mean)

    # Normalizzazione simulata
    if np.max(simulated_psd_resampled) == 0:
        simulated_psd_resampled = simulated_psd_resampled + 1e-12
    simulated_psd_resampled /= np.max(simulated_psd_resampled)

        # Normalizzo le PSD a distribuzioni di massa (somma = 1)
    real_weights = real_psd_mean.astype(float)
    sim_weights  = simulated_psd_resampled.astype(float)

    s_real = np.sum(real_weights)
    s_sim  = np.sum(sim_weights)
    if s_real == 0:
        real_weights = np.ones_like(real_weights) / len(real_weights)
    else:
        real_weights = real_weights / s_real

    if s_sim == 0:
        sim_weights = np.ones_like(sim_weights) / len(sim_weights)
    else:
        sim_weights = sim_weights / s_sim

    emd_value = wasserstein_distance(u_values=freqs_uniform,
                                    v_values=freqs_uniform,
                                    u_weights=real_weights,
                                    v_weights=sim_weights)
    
        # Plot
    plt.figure(figsize=(10,4))
    plt.plot(simulated_freqs, simulated_psd_mean, color='b', linewidth=2, label=label_sim)
    plt.plot(freqs_uniform, real_psd_mean, color='r', linewidth=2, label=label_real)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Normalized PSD")
    plt.title("PSD: Real (bar) vs Simulated (line)")
    plt.legend()
    if gv.save_plots:
        fig_name = f"sim_vs_real_PSD.png"
        os.makedirs(gv.fig_folder, exist_ok=True)
        plt.savefig(os.path.join(gv.fig_folder, fig_name), dpi=200)
    if gv.show_plots:
        plt.show()
    else:
        plt.close()

    if return_value:
        return emd_value
    
def real_vs_simulated_FC(fc_healty, time_series, return_value=False):
    """
    Compares simulated functional connectivity (from TVB) with real EEG-based FC
    at both full and reduced (macro) resolution levels.

    Displays comparison plots with MAE values.
    """

    # Extract EEG correlation matrix
    tsr_corr = ev(time_series)
    corr_eeg = tsr_corr.array_data[..., 0, 0]
    corr_eeg -= np.eye(len(time_series[0, 0, :, 0]))
    np.fill_diagonal(corr_eeg, 0.0)
    corr_eeg = np.clip(corr_eeg, -0.999999, 0.999999)

    # Align and compute macro-level FC
    sub_fc_healty, com_fc_healty = [], []
    for i in range(len(fc_healty[:, 0, 0])):
        matrix1, matrix2 = to_common_indices(corr_eeg, fc_healty[i, :, :])
        com_fc_healty.append(matrix2)
        sub_fc_healty.append(macro_fc(matrix2))
    corr_eeg = matrix1
    sub_corr_eeg = macro_fc(corr_eeg)
    com_fc_healty = np.array(com_fc_healty)
    com_fc_healty = np.clip(com_fc_healty, -0.999999, 0.999999)

    # --- Compute group-level statistics ---
    normal_corr_stack = np.stack(com_fc_healty, axis=0)
    normal_z_stack = np.arctanh(normal_corr_stack)
    normal_z_mean = np.mean(normal_z_stack, axis=0)
    normal_corr_mean = np.tanh(normal_z_mean)
    normal_mae = np.mean(np.abs(normal_corr_mean - corr_eeg))

    # Normalize matrices to [-1, 1] only for visualization purpose
    corr_eeg = 2 * (corr_eeg - np.min(corr_eeg)) / (np.max(corr_eeg) - np.min(corr_eeg)) - 1
    normal_corr_mean = 2 * (normal_corr_mean - np.min(normal_corr_mean)) / (np.max(normal_corr_mean) - np.min(normal_corr_mean)) - 1

    # Plot comparison (regular scale)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    im1 = ax[0].imshow(corr_eeg, cmap='bwr', vmin=-1, vmax=1)
    ax[0].set_title("Regular size simulated FC")
    im2 = ax[1].imshow(normal_corr_mean, cmap='bwr', vmin=-1, vmax=1)
    ax[1].set_title(f"Regular size real FC: MAE = {normal_mae:.3f}")
    plt.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)

    if gv.save_plots:
        fig_name = "Sim_vs_real_regular_size.png"
        os.makedirs(gv.fig_folder, exist_ok=True)
        plt.savefig(os.path.join(gv.fig_folder, fig_name), dpi=200)

    if gv.show_plots:
        plt.show()
    else:
        plt.close()

    sub_corr_eeg = np.clip(sub_corr_eeg, -0.999999, 0.999999)
    # --- Reduced (macro) resolution ---
    sub_fc_healty = np.array(sub_fc_healty)
    sub_fc_healty = np.clip(sub_fc_healty, -0.999999, 0.999999)
    sub_corr_stack = np.stack(sub_fc_healty, axis=0)
    sub_z_stack = np.arctanh(sub_corr_stack)
    sub_z_mean = np.mean(sub_z_stack, axis=0)
    sub_corr_mean = np.tanh(sub_z_mean)
    reduced_mae = np.mean(np.abs(sub_corr_mean - sub_corr_eeg))


    # Normalize for visualization
    sub_corr_eeg = 2 * (sub_corr_eeg - np.min(sub_corr_eeg)) / (np.max(sub_corr_eeg) - np.min(sub_corr_eeg)) - 1
    sub_corr_mean = 2 * (sub_corr_mean - np.min(sub_corr_mean)) / (np.max(sub_corr_mean) - np.min(sub_corr_mean)) - 1

    # Plot comparison (reduced scale)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    im1 = ax[0].imshow(sub_corr_eeg, cmap='bwr', vmin=-1, vmax=1)
    ax[0].set_title("Reduced size simulated FC")
    im2 = ax[1].imshow(sub_corr_mean, cmap='bwr', vmin=-1, vmax=1)
    ax[1].set_title(f"Reduced size real FC: MAE = {reduced_mae:.3f}")
    plt.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)

    if gv.save_plots:
        fig_name = "Sim_vs_real_reduced_size.png"
        os.makedirs(gv.fig_folder, exist_ok=True)
        plt.savefig(os.path.join(gv.fig_folder, fig_name), dpi=200)

    if gv.save_plots:
        fig_name = f"sim_vs_real_PSD.png"
        os.makedirs(gv.fig_folder, exist_ok=True)
        plt.savefig(os.path.join(gv.fig_folder, fig_name), dpi=200)
    if gv.show_plots:
        plt.show()
    else:
        plt.close()

    if return_value:
        return normal_mae, reduced_mae

def to_common_indices(tvb_matrix, real_matrix):
    """
    Aligns two functional connectivity matrices (from TVB simulation and real EEG data)
    by selecting only the common EEG channel indices.

    Parameters
    ----------
    tvb_matrix : np.ndarray
        TVB 65x65 functional connectivity matrix.
    real_matrix : np.ndarray
        Real EEG 62x62 functional connectivity matrix.

    Returns
    -------
    tvb_matrix_sub : np.ndarray
        Submatrix of tvb_matrix restricted to common channels.
    real_matrix_sub : np.ndarray
        Submatrix of real_matrix restricted to common channels.
    """
    your_path = 'C:/Users/User/OneDrive - University of Pisa/Desktop/TVB_tutorials/Dati_Healthy/'

    # Channel names for real EEG (62 channels)
    eeg_real_data_names = np.load(your_path + 'names_ch.npy')
    eeg_real_data_names = np.array(eeg_real_data_names)

    # Channel names for Brainstorm EEG (65 channels)
    eeg_brainstorm_65_names = np.array([
        'Fp1', 'Fp2', 'F4', 'F3', 'C3', 'C4', 'P4', 'P3', 'O2', 'O1',
        'F8', 'F7', 'T4', 'T3', 'T6', 'T5', 'Pz', 'Fz', 'IO1', 'IO2',
        'AF9', 'AF10', 'F9', 'F10', 'CB1', 'CB2', 'TP7', 'TP9', 'TP10',
        'TP8', 'Oz', 'Iz', 'PO4', 'PO3', 'CP5', 'CP6', 'CP1', 'CP2',
        'FT9', 'FT10', 'FC2', 'FC1', 'AF3', 'AF4', 'FC6', 'FC5', 'CPz',
        'P1', 'POz', 'P2', 'P6', 'C6', 'P5', 'C1', 'C2', 'C5', 'F2',
        'F6', 'F1', 'AF8', 'F5', 'AF7', 'Fpz', 'FCz', 'Cz', 'T8', 'T7',
        'P8', 'P7'
    ])

    common_names = extract_common_names()

    # Find index positions of common channels in both naming systems
    indices_65 = [np.where(eeg_brainstorm_65_names == name)[0][0] for name in common_names]
    indices_62 = [np.where(eeg_real_data_names == name)[0][0] for name in common_names]

    # Extract the aligned submatrices
    tvb_matrix_sub = tvb_matrix[np.ix_(indices_65, indices_65)]
    real_matrix_sub = real_matrix[np.ix_(indices_62, indices_62)]

    return tvb_matrix_sub, real_matrix_sub

def extract_common_names():
    """
    Finds the set of EEG channel names common between Brainstorm (65 channels)
    and the real EEG dataset (62 channels).

    Returns
    -------
    common_names : list of str
        Ordered list of channel names present in both datasets.
    """
    your_path = 'C:/Users/User/OneDrive - University of Pisa/Desktop/TVB_tutorials/Dati_Healthy/'
    eeg_real_data_names = np.load(your_path + 'names_ch.npy')
    eeg_real_data_names = np.array(eeg_real_data_names)

    eeg_brainstorm_65_names = np.array([
        'Fp1', 'Fp2', 'F4', 'F3', 'C3', 'C4', 'P4', 'P3', 'O2', 'O1',
        'F8', 'F7', 'T4', 'T3', 'T6', 'T5', 'Pz', 'Fz', 'IO1', 'IO2',
        'AF9', 'AF10', 'F9', 'F10', 'CB1', 'CB2', 'TP7', 'TP9', 'TP10',
        'TP8', 'Oz', 'Iz', 'PO4', 'PO3', 'CP5', 'CP6', 'CP1', 'CP2',
        'FT9', 'FT10', 'FC2', 'FC1', 'AF3', 'AF4', 'FC6', 'FC5', 'CPz',
        'P1', 'POz', 'P2', 'P6', 'C6', 'P5', 'C1', 'C2', 'C5', 'F2',
        'F6', 'F1', 'AF8', 'F5', 'AF7', 'Fpz', 'FCz', 'Cz', 'T8', 'T7',
        'P8', 'P7'
    ])

    eeg_real_set = set(eeg_real_data_names)
    eeg_brainstorm_set = set(eeg_brainstorm_65_names)
    common_names = eeg_real_set.intersection(eeg_brainstorm_set)

    # Keep Brainstorm order
    common_names = [name for name in eeg_brainstorm_65_names if name in eeg_real_set]
    return common_names

def compute_matrices_diff(m1, m2):
    """
    Computes the difference between two matrices and their correlation coefficient.

    Parameters
    ----------
    m1, m2 : np.ndarray
        Matrices of the same shape.

    Outputs
    -------
    Prints Pearson correlation coefficient and displays difference heatmap.
    """
    diff = m1 - m2
    mask = np.triu(np.ones(m1.shape), k=1).astype(bool)
    r, _ = pearsonr(m1[mask], m2[mask])
    print("Correlation coefficient:", r)

    mae = np.mean(np.abs(diff))
    plt.imshow(diff, cmap='bwr', vmin=-1, vmax=1)
    plt.title(f"Mean absolute difference: {mae:.3f}")
    plt.colorbar()
    plt.show()

def match_name(name, coord_names):
    """Helper function to match partial channel names to coordinate entries."""
    for c in coord_names:
        if name in c:
            return c
    raise ValueError(f"{name} not found in coordinate file.")

def quadrants_division(coord_file=r'C:\Users\User\OneDrive - University of Pisa\Desktop\TVB_Distribution\tvb_data\Lib\site-packages\tvb_data\sensors\eeg_brainstorm_65.txt'):
    """
    Divides EEG channels into 4 macroareas (quadrants) based on spatial coordinates.
    """
    common_names = extract_common_names()

    # Read coordinates
    data = []
    with open(coord_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                name, x, y, z = parts
                data.append((name, float(x), float(y), float(z)))

    coord_dict = {name: np.array([x, y, z]) for name, x, y, z in data}
    coord_names = list(coord_dict.keys())

    # Extract coordinates for common channels
    coords_common = np.array([coord_dict[match_name(name, coord_names)] for name in common_names])

    # Compute center for reference
    center = np.mean(coords_common, axis=0)
    cx, cy, cz = center

    # Assign sensors to quadrants
    quadrants = {1: [], 2: [], 3: [], 4: []}
    for i, coord in enumerate(coords_common):
        x, y, z = coord
        if x >= cx and y >= cy:
            quadrants[1].append(i)
        elif x < cx and y >= cy:
            quadrants[2].append(i)
        elif x < cx and y < cy:
            quadrants[3].append(i)
        else:
            quadrants[4].append(i)

    return quadrants

def sestants_division(coord_file=r'C:\Users\User\OneDrive - University of Pisa\Desktop\TVB_Distribution\tvb_data\Lib\site-packages\tvb_data\sensors\eeg_brainstorm_65.txt'):
    """
    Divides EEG channels into 6 macroareas (sestants) based on spatial coordinates.
    """
    common_names = extract_common_names()

    # Read coordinates
    data = []
    with open(coord_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                name, x, y, z = parts
                data.append((name, float(x), float(y), float(z)))

    coord_dict = {name: np.array([x, y, z]) for name, x, y, z in data}
    coord_names = list(coord_dict.keys())
    coords_common = np.array([coord_dict[match_name(name, coord_names)] for name in common_names])

    # Compute coordinate ranges
    width_y = np.abs(np.max(coords_common[:, 1]) - np.min(coords_common[:, 1]))
    cx = np.abs(np.max(coords_common[:, 0]) + np.min(coords_common[:, 0])) / 2

    sestants = {i: [] for i in range(1, 7)}
    for i, coord in enumerate(coords_common):
        x, y, z = coord
        if x >= cx and y >= (width_y * 0.67 + np.min(coords_common[:, 1])):
            sestants[1].append(i)
        elif x >= cx and (width_y/3 + np.min(coords_common[:, 1])) <= y < (width_y * 0.67 + np.min(coords_common[:, 1])):
            sestants[2].append(i)
        elif x >= cx and y < (width_y/3 + np.min(coords_common[:, 1])):
            sestants[3].append(i)
        elif x < cx and y >= (width_y * 0.67 + np.min(coords_common[:, 1])):
            sestants[4].append(i)
        elif x < cx and (width_y/3 + np.min(coords_common[:, 1])) <= y < (width_y * 0.67 + np.min(coords_common[:, 1])):
            sestants[5].append(i)
        else:
            sestants[6].append(i)
    return sestants

def macro_fc(matrix):
    """
    Computes a macro-scale functional connectivity matrix by averaging
    FC values over predefined brain regions (quadrants or sestants).

    Returns
    -------
    macro_fc : np.ndarray
        Aggregated matrix of shape (gv.partitioning, gv.partitioning)
    """
    if gv.partitioning == 4:
        macroareas = quadrants_division()
    elif gv.partitioning == 6:
        macroareas = sestants_division()
    else:
        raise ValueError("Unsupported number of macroareas.")

    macro_fc = np.zeros((gv.partitioning, gv.partitioning))
    for i in range(1, gv.partitioning + 1):
        for j in range(1, gv.partitioning + 1):
            idx_i = macroareas[i]
            idx_j = macroareas[j]
            submatrix = matrix[np.ix_(idx_i, idx_j)]
            macro_fc[i - 1, j - 1] = np.mean(submatrix)
    return macro_fc

def analyze_regular_data(real_data):
    """
    Computes the mean and standard deviation of real EEG FC matrices
    in both correlation and Fisher z-space.
    """
    normal_matrices = np.array(real_data)
    normal_corr_stack = np.stack(normal_matrices, axis=0)
    normal_z_stack = np.arctanh(normal_corr_stack)
    normal_z_mean = np.mean(normal_z_stack, axis=0)
    normal_z_std = np.std(normal_z_stack, axis=0)
    mean_std = np.mean(normal_z_std)

    normal_corr_mean = np.tanh(normal_z_mean)
    normal_corr_mean = 2 * (normal_corr_mean - np.min(normal_corr_mean)) / (np.max(normal_corr_mean) - np.min(normal_corr_mean)) - 1

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    im1 = ax[0].imshow(normal_corr_mean, cmap='bwr', vmin=-1, vmax=1)
    ax[0].set_title("Regular size mean FC")
    im2 = ax[1].imshow(normal_z_std, cmap='viridis')
    ax[1].set_title(f"Regular size FC std (z-space): mean std = {mean_std:.3f}")
    plt.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)

    if gv.save_plots:
        fig_name = "Regular_size_real_data.png"
        os.makedirs(gv.fig_folder, exist_ok=True)
        plt.savefig(os.path.join(gv.fig_folder, fig_name), dpi=200)

    plt.show()

def analyze_reduced_data(real_data):
    """
    Resizes both standard and real EEG FC matrices and computes the mean
    and standard deviation in both correlation and Fisher z-space.
    """
    sub_matrices = np.zeros((len(real_data), gv.partitioning, gv.partitioning))
    
    for i in range(len(real_data)):
        sub_matrices[i, :, :] = macro_fc(matrix=real_data[i, :, :])
    
    # Shift to z-score to compute mean and standard deviation
    # Stack into a 3D array: shape = (n_subjects, n_regions, n_regions)
    sub_corr_stack = np.stack(sub_matrices, axis=0)
    # --- Fisher z-transform ---
    sub_z_stack = np.arctanh(sub_corr_stack)
    # --- Compute mean and std in z-space ---
    sub_z_mean = np.mean(sub_z_stack, axis=0)
    sub_z_std  = np.std(sub_z_stack, axis=0)
    sub_mean_std = np.mean(sub_z_std)
    # --- Convert mean back to correlation space ---
    sub_corr_mean = np.tanh(sub_z_mean)
    sub_corr_mean = 2 * (sub_corr_mean - np.min(sub_corr_mean)) / (np.max(sub_corr_mean) - np.min(sub_corr_mean)) - 1    

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    im1 = ax[0].imshow(sub_corr_mean, cmap='bwr', vmin=-1, vmax=1)
    ax[0].set_title("Reduced size mean FC")
    im2 = ax[1].imshow(sub_z_std, cmap='viridis')
    ax[1].set_title("Reduced size FC std (z-space): mean std = " + str(sub_mean_std))
    plt.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)

    if gv.save_plots:
        fig_name = "Reduced_size_real_data.png"
        os.makedirs(gv.fig_folder, exist_ok=True)
        plt.savefig(os.path.join(gv.fig_folder, fig_name), dpi=200)

    plt.show()

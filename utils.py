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
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
from scipy import signal

from classes import JRPSP
from global_variable_creation import get_eeg_monitor, get_conn
import global_variable_creation as gv


def skin_and_sensors_visualization():
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

def brain_activity_visualization(eeg, tavg, start_time=-1000):
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

def eeg_visualization(datas, times, display_time=3000, chlist=[4, 13, 20, 23, 38, 41, 45, 50, 62]):
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

def simulate(sim_time=2000, noise=50e-3, cp=24, np_parameter=3,
             g=1, velocity=np.inf, cip=0, cep=1, taue=0,
             tau_e_values=np.linspace(0.294, 0.2945, 2),
             C_ip_values=np.linspace(35, 10, 2),
             C_ep_values=np.linspace(110, 20, 2),
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
    a_value = 0.0325 / tau_e_values[taue]
    J_value = 128

    model_pars = dict(
        A=3.25, B=22, v0=6.0,
        a=a_value, b=0.05, r=0.56,
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
        coupling=tvb.coupling.Linear(a=np.r_[a_value * 10]),
        integrator=tvb.integrators.EulerStochastic(dt=0.1, noise=noise),
        stimulus=stimulus,
        monitors=(mon_tavg, mon_eeg),
        simulation_length=sim_time
    )
    sim.configure()

    (ttavg, tavg), (teeg, eeg) = sim.run(simulation_length=sim_time)
    return ttavg, tavg, teeg, eeg


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

def psd(time_series, label, lowcut=0.5, highcut=45, n_seg=2048, window='hamming'):
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

def NSGA_simulate(sim_time, cp=0, np_parameter=3,
             g=1,
             velocity=1,
             noise=1,
             a=0,
             b=0,
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
    print(">>> gv module path:", gv.__file__)
    print(">>> gv.temp_avg_period =", getattr(gv, "temp_avg_period", "NON DEFINITO"))
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


def NSGA_psd(time_series, lowcut=0.5, highcut=45, n_seg=2048, window='hamming'):
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

    # Normalize
    psd_mean /= np.max(psd_mean)

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
    
def plot_real_vs_simulated_psd_bar(path, time_series, n_bin=180, fs=100,
                                   label_real="Experimental", label_sim="Simulated"):

    # =========================
    # PSD reale
    # =========================
    matrix = np.load(path)
    real_psd_mean = np.mean(matrix, axis=0)

    # Normalizzazione reale
    real_psd_mean = real_psd_mean[:n_bin]
    real_psd_mean /= np.max(real_psd_mean)

    # Frequenze reali (uniformi)
    freqs_uniform = np.linspace(0, 45, n_bin)

    # =========================
    # PSD simulata
    # =========================
    simulated_freqs, simulated_psd_mean = NSGA_psd(time_series=time_series)

    # Taglio frequenze a 45 Hz
    mask = simulated_freqs <= 45
    simulated_freqs = simulated_freqs[mask]
    simulated_psd_mean = simulated_psd_mean[mask]

    # =========================
    # Interpolazione simulata su 180 bin
    # =========================
    simulated_psd_resampled = np.interp(freqs_uniform, simulated_freqs, simulated_psd_mean)

    # Normalizzazione simulata
    simulated_psd_resampled /= np.max(simulated_psd_resampled)

    psd_mae = np.sum(np.abs(simulated_psd_resampled-real_psd_mean))

    # =========================
    # Plot a barre affiancate
    # =========================
    width = (freqs_uniform[1] - freqs_uniform[0]) * 0.4  # due barre affiancate

    """
    plt.figure(figsize=(12, 6))

    plt.bar(freqs_uniform - width/2, real_psd_mean,
            width=width, label=label_real, alpha=0.7)

    plt.bar(freqs_uniform + width/2, simulated_psd_resampled,
            width=width, label=label_sim, alpha=0.7)

    plt.title('PSD Comparison (bin-by-bin)', fontsize=20)
    plt.ylim(0, 1.1)
    plt.xlim(0, 45)
    plt.xlabel('Frequency [Hz]', fontsize=16)
    plt.ylabel('Normalized PSD', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)

    plt.show()
    """
    return psd_mae


def plot_real_psd_bar(path, time_series, n_bin=180, fs=100, label_real="Experimental", label_sim="Simulated"):

    # PSD simulata
    simulated_freqs, simulated_psd_mean = NSGA_psd(time_series=time_series)

    # Plot
    plt.figure(figsize=(10,4))
    plt.plot(simulated_freqs, simulated_psd_mean, color='b', linewidth=2, label=label_sim)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Normalized PSD")
    plt.title("PSD: Real (bar) vs Simulated (line)")
    plt.legend()
    plt.show()

#ttavg, tavg, teeg, eeg = simulate(sim_time = 20000, C_ep_values=[100.0], C_ip_values=[75.0], tau_e_values=[0.1], cip=0, cep=0,taue=0)
#ttavg, tavg = preprocess(ttavg, tavg)
#teeg, eeg = preprocess(teeg, eeg)

#plot_real_vs_simulated_psd_bar(path=r"C:\Users\User\OneDrive - University of Pisa\Desktop\TVB_tutorials\Dati_Healthy\psd_ctr_preview.npy", time_series=eeg)
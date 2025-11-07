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
from global_variable_creation import get_eeg_monitor, get_conn, show_plots, save_plots, th, temp_avg_period, fig_folder


def skin_and_sensors_visualization():
    conn = get_conn()

    # Patient's skin configuration
    skin = surfaces.SkinAir.from_file()
    skin.configure()

    # EEG sensors configuration
    sens_eeg = sensors.SensorsEEG.from_file()
    sens_eeg.configure()

    plt.figure(figsize = (6,6))
    ax = plt.subplot(111, projection='3d')

    # Plot cortical regions as black dots:
    x, y, z = conn.centres.T[:,:]
    ax.plot(x, y, z, 'ko')

    # Plot EEG sensors as blue x's:
    x, y, z = sens_eeg.sensors_to_surface(skin).T
    ax.plot(x, y, z, 'rx')

    # Plot boundary surface
    x, y, z = skin.vertices.T
    ax.plot_trisurf(x, y, z, triangles=skin.triangles, alpha=0.1, edgecolor='k')

    if save_plots == True:
        fig_name = f"Skin_and_sensors.png"
        os.makedirs(fig_folder, exist_ok=True)
        fig_path = os.path.join(fig_folder, fig_name)
        plt.savefig(fig_path, dpi=200)

    if show_plots == True:
        plt.show()

    plt.figure()
    plt.imshow(conn.weights, cmap = 'binary')

    if save_plots == True:
        fig_name = f"Connectivity.png"
        fig_path = os.path.join(fig_folder, fig_name)
        plt.savefig(fig_path, dpi=200)

    if show_plots == True:
        plt.show()

def brain_activity_visualization(eeg, tavg, start_time = -1000):

    conn = get_conn()
    mon_eeg = get_eeg_monitor()

    # Dimensions of eeg and tavg are [time, channels]
    eeg = eeg[start_time:, :]
    eeg /= (np.max(eeg,0) - np.min(eeg,0 )) #normalization step
    eeg -= np.mean(eeg, 0)  #re-referencing step
    times_eeg = np.linspace(0,np.shape(eeg)[0], np.shape(eeg)[0]) #creating an array of time-steps

    tavg = tavg[start_time:]
    tavg /= (np.max(tavg,0) - np.min(tavg,0 )) #normalization step
    tavg -= np.mean(tavg, 0)  #re-referencing step
    times_tavg = np.linspace(0,np.shape(tavg)[0], np.shape(tavg)[0]) #creating an array of time-steps

    plt.figure(figsize=(10,12))
    plt.plot(times_eeg, eeg + np.r_[:len(mon_eeg.sensors.labels)], color = 'b') # the np.r_[:76] is addedd to separate channels or regions
    plt.yticks(np.r_[:len(mon_eeg.sensors.labels)], mon_eeg.sensors.labels, fontsize = 12)
    plt.title("Brain activity (EEG) ", fontsize = 18)
    plt.xlabel('time (ms)', fontsize = 16)

    if save_plots == True:
        fig_name = f"Brain_activity_EEG.png"
        os.makedirs(fig_folder, exist_ok=True)
        fig_path = os.path.join(fig_folder, fig_name)
        plt.savefig(fig_path, dpi=200)

    if show_plots == True:
        plt.show()

    plt.figure(figsize=(10,12))
    plt.plot(times_tavg, tavg + np.r_[:len(conn.region_labels)], color = 'red') # the np.r_[:76] is addedd to separate channels or regions
    plt.yticks(np.r_[:len(conn.region_labels)], conn.region_labels, fontsize = 12)
    plt.title("Brain activity (region space) ", fontsize = 18)
    plt.xlabel('time (ms)', fontsize = 16)

    if save_plots == True:
        fig_name = f"Brain_activity_region_spece.png"
        os.makedirs(fig_folder, exist_ok=True)
        fig_path = os.path.join(fig_folder, fig_name)
        plt.savefig(fig_path, dpi=200)

    if show_plots == True:
        plt.show()

def eeg_visualization(datas, times, display_time = 3000, chlist = [4,13,20,23,38,41,45,50,62]):

    mon_eeg = get_eeg_monitor()

    for key, value in datas.items():
        time_series = value['eeg']
        label = value['label']
        color = value['color']

        #Plot EEG
        plt.figure(figsize=(10,12))
        plt.plot(times[:display_time], time_series[:display_time, 0, chlist, 0] + np.r_[:(len(chlist))], color = color)
        plt.yticks(np.r_[:len(chlist)], mon_eeg.sensors.labels[chlist], fontsize = 12)
        plt.title("EEG sensors, " + label, fontsize = 18)
        plt.xlabel('time (ms)', fontsize = 16)

        if save_plots == True:
            fig_name = f"EEG_sensors_" + label + f".png"
            os.makedirs(fig_folder, exist_ok=True)
            fig_path = os.path.join(fig_folder, fig_name)
            plt.savefig(fig_path, dpi=200)

        if show_plots == True:
            plt.show()

def simulate(sim_time, structural_connectivities, noise = 50e-3, cp = 24, np_parameter = 3, g = 1, velocity = np.inf, cip = 1, cep = 1, taue = 1, tau_e_values = np.linspace(0.270,0.310,2), C_ip_values = np.linspace(35,10,2), C_ep_values = np.linspace(110,20,2), impaired_regions = [], stimulus = None):
    '''This function simulates signals at different levels of neurodegeneration parameters. The regions in which hypoinhibition is modelled are passed 
    through the "regions" argument which by default is equal to the "impaired_regions" listed above''' 
    conn = get_conn()
    conn.weights = structural_connectivities[cp, np_parameter, :, :] #This alters the connectome according to equation 3 for a combination of cp and np
    # cp is comprised between 0 and 2 with 50 equal steps, while neuroplasticity is uniformally distributed between 0 and 2 with steps of 0.25 amplitude (plus a value of 2.5 at the end)
    conn.weights = conn.weights / np.max(conn.weights)              #normalised by maximum weight  
    conn.weights *= g
    conn.speed = np.array(velocity)                               #To select conduction velocity of the signal
    conn.configure()
    
    Cep_value = C_ep_values[cep] #this determines the value of C_ep = J*a_2
    Cip_value = C_ip_values[cip] #this determines the value of C_ip = J*a_4
    a_value = 0.0325/tau_e_values[taue] #this determines the value of t_e = 1/a; 0.032.5 is a constant determined
    #so that when H_e is 3.25 mV then tau_e is 10 ms
    J_value = 128

    # setup model based on paper's parameters
    model_pars = dict(
        A=3.25, #excitatory PSP
        B=22, #inhibitory PSP
        v0=6.0,
        a=a_value, #a_value, # increased with respect to the healthy case. (also remember that TVB uses ms, not s)
        b=0.05,    # decreased with respect to the healthy case. (also remember that TVB uses ms, not s)
        r=0.56,
        nu_max=0.0025, # e0 in the JR original paper
        # TVB factors C_i into J*a_i, e.g. C1 = a_1 * J
        J= J_value,
        a_1=1.0,
        a_2=Cep_value/J_value,
        a_3=0.25,
        a_4=Cip_value/J_value,
        mu=0.22, # baseline input, avg of 120 and 320 pulses/s
    )

    # implement JR + afferent PSP for setting the JR in the right portion of the phase space
    # factor out noise from dy4 = A a (p(t) + ...) as y4 += dt (...) + A a dW_t
    # this allows us to model the noise as TVB does it, though scaling requires experiment
    nsig = np.zeros((8, 76, 1))
    nsig[4] = model_pars['A'] * model_pars['a'] * (.320 - .120) * noise
    noise = tvb.noise.Additive(nsig=nsig)

    #setting up monitors:
    mon_eeg = get_eeg_monitor()
    mon_tavg = monitors.TemporalAverage(period=temp_avg_period)
    #Bundling
    what_to_watch = (mon_tavg, mon_eeg)
    sim = simulator.Simulator(connectivity=conn,
                         conduction_speed=float(conn.speed),
                         model=JRPSP(
                         variables_of_interest=("y0", "y1", "y2", "y3", "y4", "y5", "y6", "y7"),
                         **{k: np.array(v) for k, v in model_pars.items()}),
                         coupling=tvb.coupling.Linear(a=np.r_[a_value*10]),
                         integrator=tvb.integrators.EulerStochastic(dt=0.1, noise=noise), 
                         stimulus = stimulus,
                         monitors=what_to_watch,
                         simulation_length=sim_time)
    sim.configure()

    (ttavg, tavg), (teeg, eeg) = sim.run(simulation_length=sim_time) #tavg is the name of signals in region space

    return ttavg, tavg, teeg, eeg # ttavg and teeg are the timestamps

def preprocess(ttavg, tavg, teeg, eeg, PSD = True, normalize = True, cut = 1000):

    if PSD:
        #Discarding initial transient for PSD analysis
        ttavg = ttavg[cut:]
        tavg = tavg[cut:, :, :, :]
        teeg = teeg[cut:]
        eeg = eeg[cut:, :, :, :]        
        ttavg -= cut
        teeg -= cut

    if normalize:
        #Normalizing and subtracting average
        tavg /= (np.max(tavg,0) - np.min(tavg,0 ))
        tavg -= np.mean(tavg, 0)
        eeg /= (np.max(eeg,0) - np.min(eeg,0 ))
        eeg -= np.mean(eeg, 0)
                           
    return ttavg, tavg, teeg, eeg

def ev(time_series):
    
    conn = get_conn()
    tsr = TimeSeriesRegion(connectivity=conn,
                       data=time_series,                            #in TVB 4D format
                       sample_period=temp_avg_period) #in ms
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
    tsr_corr = ev(time_series)
    corr = tsr_corr.array_data[..., 0, 0]
    corr -= np.eye(len(time_series[0,0,:,0]))
    #Define a threshold for connections
    corr_th = (corr) > th
    pmatrix = np.zeros((len(time_series[0,0,:,0]),len(time_series[0,0,:,0])))

    for i in range(len(time_series[0,0,:,0])):
        for j in range(len(time_series[0,0,:,0])):
            region_A = time_series[:,0, i, 0]
            region_B = time_series[:,0, j, 0]
            pmatrix[i][j] = pearsonr(region_A, region_B)[1]

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

    if save_plots == True:
        fig_name = label + f"_FC_.png"
        os.makedirs(fig_folder, exist_ok=True)
        fig_path = os.path.join(fig_folder, fig_name)
        plt.savefig(fig_path, dpi=200)

    if show_plots == True:
        plt.show()

def functional_connectivity_alpha(time_series, label, lowcutfreq = 8, highcutfreq = 13):
    # fs is the sampling frequency (1000 Hz here)
   
    # Apply bandpass filter to the time series (8-13 Hz for alpha band)
    filtered_time_series = np.zeros_like(time_series)
    
    for i in range(time_series.shape[2]):
        for j in range(time_series.shape[3]):
            filtered_time_series[:, 0, i, j] = bandpass_filter(time_series[:, 0, i, j], lowcutfreq, highcutfreq,  order=5)

    functional_connectivity(filtered_time_series, label + ' filtered (' + str(lowcutfreq) + '-' + str(highcutfreq) + ')')

def bandpass_filter(data, lowcut, highcut, order):
    nyquist = 0.5 / (temp_avg_period/1000)
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=0)
    return y

def psd(time_series, label, lowcut = 0.5, highcut = 45, n_seg = 2048, window = 'hamming'): 

    n_timepoints = time_series.shape[0]
    n_regions = time_series.shape[2]
    # Inizializza matrice per segnali filtrati
    filtered_signals = np.zeros((n_timepoints, n_regions))
    # Applica filtro passa-banda a ogni regione
    for i in range(n_regions):
        region_signal = time_series[:, 0, i, 0]
        filtered_signals[:, i] = bandpass_filter(region_signal, lowcut, highcut, order=4)

    # Calcola PSD con il metodo di Welch per ogni regione
    psd_all = []
    for i in range(n_regions):
        freqs, psd = signal.welch(
            filtered_signals[:, i],
            fs=1/(temp_avg_period/1000),
            nperseg=n_seg,
            noverlap=n_seg // 2,
            window=window
        )
        psd_all.append(psd)
    
    psd_all = np.array(psd_all)  # (n_regions, n_freqs)

    # Media e deviazione standard tra regioni
    psd_mean = np.mean(psd_all, axis=0)
    psd_std = np.std(psd_all, axis=0)

    # Normalizzazione
    psd_mean /= np.max(psd_mean)
    psd_std /= np.max(psd_mean)

    plt.figure(figsize = (10,6))
    plt.plot(freqs, psd_mean, label = label, linewidth = 4)
    plt.fill_between(freqs, psd_mean + psd_std, psd_mean - psd_std, where = None, alpha = 0.35)
    plt.title(label + 'mean PSD', fontsize = 29)
    plt.ylim(0, 1.01)
    plt.xlim(-0.1, 45)
    plt.xlabel('frequency [Hz]', fontsize = 20)
    plt.ylabel('PSD channels', fontsize = 20)
    plt.title('PSD')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(fontsize = 20)

    if save_plots == True:
        fig_name = f"PSD.png"
        os.makedirs(fig_folder, exist_ok=True)
        fig_path = os.path.join(fig_folder, fig_name)
        plt.savefig(fig_path, dpi=200)

    if show_plots == True:
        plt.show()

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


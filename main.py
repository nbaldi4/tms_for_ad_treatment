import numpy as np
from tvb.simulator.lab import *

from utils import skin_and_sensors_visualization, brain_activity_visualization, simulate, preprocess, eeg_visualization
from utils import functional_connectivity, psd, functional_connectivity
from global_variable_creation import get_eeg_monitor, get_conn


""" L'idea è di scrivere un codice che possa replicare e confrontare segnali neurali stimolati e non dati una serie di parametri, l'interfaccia servirà quindi
solo a inserire i parametri d'interesse e a scegliere se visualizzare o meno alcuni plots. Proprio a questo scopo ho messo degli hastag accanto agli elementi
chr voglio conservare in qualità di comandi per interfaccia"""
eeg_vis = True
psd_vis = True
fc_vis = True
brain_activity = True
skin_and_sensors = True
FC_alpha_visualizing = False ####################
stim_visualizing = False ########################
global_coupling_val = [1] #######################
velocity_val = [np.inf] #########################
noise_val = [50e-3] #############################
time_index = 1000 ###############################
skip_to_stim = False ############################

"""Connettività struttura e dati sani esterni sono utili e quindi lasciamo la possibilità di inserirli
ma dovremo implementare un'interfaccia adeguata"""

struct_path = 'C:/Users/User/OneDrive - University of Pisa/Desktop/TVB_tutorials/'
struct_conn = np.load(struct_path + 'structural_connectivities.npy')

psd_path = 'C:/Users/User/OneDrive - University of Pisa/Desktop/TVB_tutorials/Dati_Healthy/'
matrix_path = psd_path + 'psd_ctr_preview.npy'

# Define impaired regions (taken from Braak stages)
impaired_regions = [21,22,30,31,32,34,59,60,68,69,70,72]

# Here EEG monitor configuration can globally modified
mon = get_eeg_monitor()
conn = get_conn()

ttavg, tavg, teeg, eeg = simulate(impaired_regions = impaired_regions, structural_connectivities = struct_conn, cp = 15, np_parameter = 3, g=1, velocity=np.inf, noise=50e-3,  #This is the standard value of neuroplasticity used in the paper
                                                sim_time = 20000, C_ep_values=[110.0], C_ip_values=[35.0], tau_e_values=[0.294], cip=0, cep=0,taue=0)
ttavg, tavg, teeg, eeg = preprocess(ttavg, tavg, teeg, eeg)
datas = {'data_healthy': {'tavg': tavg, 'eeg': eeg, 'label': 'Healthy', 'color': 'blue'}}

if skin_and_sensors == True:
    skin_and_sensors_visualization()

if brain_activity == True:
    brain_activity_visualization(eeg=eeg[:,0,:,0], tavg=tavg[:,0,:,0])

if eeg_vis == True:
    eeg_visualization(datas = datas, times = teeg)

"""il controllo sulla label della FC va migliorato, vorrei si potesse inserire da terminale, magari facendo si che alla tupla di parametri richiesti 
venga associata una certa dicita (e.g. malato braak stage 2 - beta stage 1)"""
if fc_vis == True:
    functional_connectivity(eeg, 'EEG')
    functional_connectivity(tavg, 'TAVG')

if psd_vis == True:
    psd(eeg, 'EEG')
    psd(tavg, 'TAVG')

# In TVB the output for the JR (tavg or eeg) is a 4D object of dimensions [num_timestamps, num_modes, channels (or regions), 1] 
# The interesting variables are stored in the first and third argument, in which you find the timeseries of channels (if you refer to eeg)
# or the timeseries of regions (if you refer to tavg)

#FC_alpha_visualization(datas, FC_alpha_visualizing, mon_EEG, conn, period=period)

#compare_matrix_and_computed_psd(path = matrix_path, values = eeg_healthy, values2 = eeg_very_healty, visualizing = True)

"""
stim_weight = 1
scale_fact = 10
start_time = 500

train1 = make_train([35,73], 5.0/stim_weight, onset=start_time , T=1000.0, tau=40.0) #500
train2 = make_train([36,74], 4.5/stim_weight, onset=start_time + 10 * scale_fact, T=1000, tau=50.0) #600
train3 = make_train([34,72], 4.0/stim_weight, onset=start_time + 35 * scale_fact, T=1000, tau=80.0) #850
train4 = make_train([32,70], 4.0/stim_weight, onset=start_time + 40 * scale_fact, T=1000, tau=80.0) #900
train5 = make_train([24,62], 4.0/stim_weight, onset=start_time + 50 * scale_fact, T=1000, tau=80.0) #1000
stimulus = MultiStimuliRegion(train1, train2,train3, train4, train5)
stimulus.configure_space()
time = np.r_[1e3:3e3:15.0]
stimulus.configure_time(time)
pattern = stimulus()


sim_time = 4000
display_time = sim_time - 2000
ttavg_stim, tavg_healthy_stim, teeg_stim, eeg_healthy_stim = Simulate(lp=15, cp=26, impaired_regions=impaired_regions, structural_connectivities = struct_conn, np_parameter = 3, g=1, velocity=np.inf, noise=50e-3, 
                                                stimulus = stimulus, sim_time = sim_time) #Less simulated time needed
ttavg, tavg_healthy, teeg, eeg_healthy = preprocess(ttavg_stim, tavg_healthy_stim, teeg_stim, eeg_healthy_stim)

ttavg_stim, tavg_MCI_stim, teeg_stim, eeg_MCI_stim = Simulate(lp=24, cp=48, impaired_regions=impaired_regions, structural_connectivities = struct_conn, np_parameter = 3, g=1, velocity=np.inf, noise=50e-3, 
                                                stimulus = stimulus, sim_time = sim_time) #Less simulated time needed
ttavg_stim, tavg_MCI_stim, teeg_stim, eeg_MCI_stim = preprocess(ttavg_stim, tavg_MCI_stim, teeg_stim, eeg_MCI_stim)

datas_stim = {'data_healthy': {'tavg': tavg_healthy_stim, 'eeg': eeg_healthy_stim, 'label': 'Healthy', 'color': 'blue'},
         'data_MCI': {'tavg': tavg_MCI_stim, 'eeg': eeg_MCI_stim,  'label': 'MCI', 'color': 'red'}}

chlist_occipital = [30,32,33,8,9]

for key, value in datas_stim.items():
    tavg_stim = value['tavg']
    eeg_stim = value['eeg']
    label = value['label']
    color = value['color']

    #Plot EEG
    plt.figure(figsize=(10,12))
    plt.plot(teeg_stim[:display_time], eeg_stim[:display_time, 0, chlist_occipital, 0] + np.r_[:(len(chlist_occipital))], color = color)
    plt.yticks(np.r_[:len(chlist_occipital)], mon_EEG.sensors.labels[chlist_occipital], fontsize = 12)
    plt.title("EEG occipital sensors," + label, fontsize = 18)
    plt.xlabel('time (ms)', fontsize = 16)
    plt.show()

    
################################## USEFUL ONLY WHEN STIMULUS IS APPLIED ###################################################

#1500 is stimulus onset 
onset_stim = 1500
mask1 = onset_stim + 60
mask2  = onset_stim + 150

chlist_occipital = [30,32,33,8,9]

chlist = chlist_occipital

stimulus_visualization(stim_visualizing, onset_stim, datas_stim, teeg_stim, mask1, mask2, chlist)
"""
import os

import numpy as np
import argparse
from tvb.simulator.lab import *

from utils import skin_and_sensors, brain_activity, simulate, preprocess, eeg_plot
from utils import functional_connectivity, psd, functional_connectivity_band
from data_analysis_utils import real_vs_simulated_FC, real_vs_simulated_PSD, NSGAII, analyze_regular_data, analyze_reduced_data, load_simulation_results
import global_variable_creation as gv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_plots", action="store_true", help="Show plots on screen")
    parser.add_argument("--no_sim", action="store_true", help="Search for simulated data")
    parser.add_argument("--save_plots", action="store_true", help="Save plots to disk")
    parser.add_argument("--save_folder", type=str, help="Folder to save figures and data")
    parser.add_argument("--struct_conn", type=str, help="Path to structural connectivity file")
    parser.add_argument("--EEG", action="store_true", help="Perform eeg")
    parser.add_argument("--PSD", action="store_true", help="Perform psd")
    parser.add_argument("--FC", action="store_true", help="Perform fc")
    parser.add_argument("--brain_act", action="store_true", help="Compute brain activity")
    parser.add_argument("--skin_and_sens", action="store_true", help="Visualize skin and sensors positions")
    parser.add_argument("--th", type=float, help="Threshold for FC visualization")
    parser.add_argument("--temp_avg_period", type=float, help="Average integration step")
    parser.add_argument("--sim_vs_real", action="store_true", help="Perform comparison between simulation and real data")
    parser.add_argument("--g", type=float, help="Coupling")
    parser.add_argument("--vel", type=float, help="Conduction velocity")
    parser.add_argument("--noise", type=float, help="Noise")
    parser.add_argument("--a", type=float, help="JR a parameter")
    parser.add_argument("--b", type=float, help="JR b parameter")
    parser.add_argument("--tau", type=int, help="Tau stage (from 0 to 5)")
    parser.add_argument("--abeta", type=int, help="Beta Amyloid stage (from 0 to 5)")
    parser.add_argument("--st", type=int, help="simulation time")
    parser.add_argument("--best_mod", action="store_true", help="Model parameters optimization to maximize simulation similarity with data")
    parser.add_argument("--fact", type=float, help="diminutive factor in a values")
    parser.add_argument(
        "--impaired_regions",
        nargs="+",               # accepts one or more values
        type=int,                # convert them to integers
        help="List of impaired region indices"
    )
################

################
    args = parser.parse_args()
    gv.show_plots = args.show_plots
    gv.save_plots = args.save_plots
    gv.eeg = args.EEG
    gv.psd = args.PSD
    gv.fc = args.FC
    gv.brain_activity = args.brain_act
    gv.skin_and_sensors = args.skin_and_sens
    gv.sim_vs_real = args.sim_vs_real

    if args.g:
        gv.coup = args.g

    if args.fact:
        gv.factor = args.fact

    if args.st:
        gv.sim_time = args.st
 
    if args.vel:
        gv.vel = args.vel
    
    if args.noise:
        gv.noi = args.noise
    
    if args.a:
        gv.aa = args.a
    
    if args.b:
        gv.bb = args.b

    if args.th:
        gv.th = args.th
    else:
        gv.th = 0.08

    if args.tau:
        gv.tau_stage = args.tau
    else:
        gv.tau_stage = 0
    
    if args.abeta:
        gv.abeta_stage = args.abeta
    else:
        gv.abeta_stage = 0

    if args.temp_avg_period:
        gv.temp_avg_period = args.temp_avg_period

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

    if args.struct_conn:
        gv.struct_path = args.struct_conn

    if args.impaired_regions:
        gv.impaired_regions = args.impaired_regions

    """ L'idea è di scrivere un codice che possa replicare e confrontare segnali neurali (stimolati e non) dati una serie di parametri, l'interfaccia servirà quindi
    solo a inserire i parametri d'interesse e a scegliere se visualizzare o meno alcuni plots. Proprio a questo scopo ho messo degli hastag accanto agli elementi
    che voglio conservare in qualità di comandi per interfaccia"""
    time_index = 1000 ###############################
    skip_to_stim = False ############################
    if args.no_sim:
        sim_id = f"tau{args.tau}_abeta{args.abeta}_a{args.a:.4f}_b{args.b:.4f}_g{args.g:.4f}_v{args.vel:.2f}_n{args.noise:.4f}_simtime{args.st}"
        # 2. data loading
        ttavg, tavg, teeg, eeg = load_simulation_results(sim_id)
        ttavg, tavg = preprocess(ttavg, tavg)
        teeg, eeg = preprocess(teeg, eeg)

    if not args.no_sim:
        ttavg, tavg, teeg, eeg = simulate(g=gv.coup, velocity=gv.vel, noise=gv.noi, a=gv.aa, b=gv.bb,tau_stage=gv.tau_stage, abeta_stage=gv.abeta_stage, factor=gv.factor)#simulate(g=gv.coup, velocity=gv.vel, noise=gv.noi, a=gv.aa, b=gv.bb, tau_stage=gv.tau_stage, abeta_stage=gv.abeta_stage)
        ttavg, tavg = preprocess(ttavg, tavg)
        teeg, eeg = preprocess(teeg, eeg)

    datas = {'data_healthy': {'tavg': tavg, 'eeg': eeg, 'label': 'Healthy', 'color': 'blue'}}

    if gv.skin_and_sensors == True:
        skin_and_sensors()

    if gv.brain_activity == True:
        brain_activity(eeg=eeg[:,0,:,0], tavg=tavg[:,0,:,0])

    if gv.eeg == True:
        eeg_plot(datas = datas, times = teeg)

    """il controllo sulla label della FC va migliorato, vorrei si potesse inserire da terminale, magari facendo si che alla tupla di parametri richiesti 
    venga associata una certa dicita (e.g. malato braak stage 2 - beta stage 1)"""
    if gv.fc == True:
        functional_connectivity_band(eeg, 'EEG alpha band (8-14)Hz', lowcut=8, highcut=14)
        functional_connectivity_band(eeg, 'EEG theta band (4-8)Hz', lowcut=4, highcut=8)
        functional_connectivity_band(eeg, 'EEG delta band (0.5-4)Hz', lowcut=0.5, highcut=4)
        functional_connectivity_band(tavg, 'TAVG alpha band (8-14)Hz', lowcut=8, highcut=14)
        functional_connectivity_band(tavg, 'TAVG theta band (4-8)Hz', lowcut=4, highcut=8)
        functional_connectivity_band(tavg, 'TAVG delta band (0.5-4)Hz', lowcut=0.5, highcut=4)
        functional_connectivity(eeg, 'EEG')
        functional_connectivity(tavg, 'TAVG')

    if gv.psd == True:
        psd(eeg, 'EEG')
        psd(tavg, 'TAVG')

    if gv.real_data == True:
        analyze_regular_data(np.load(gv.real_FC_data))
        analyze_reduced_data(np.load(gv.real_FC_data))

    if gv.sim_vs_real == True:
        real_vs_simulated_FC(np.load(gv.HC_FC_data), eeg)
        real_vs_simulated_PSD(np.load(gv.HC_PSD_data), eeg)

    if args.best_mod == True:
        NSGAII()

    # In TVB the output for the JR (tavg or eeg) is a 4D object of dimensions [num_timestamps, num_modes, channels (or regions), 1] 
    # The interesting variables are stored in the first and third argument, in which you find the timeseries of channels (if you refer to eeg)
    # or the timeseries of regions (if you refer to tavg)

    #FC_alpha_visualization(datas, FC_alpha_visualizing, mon_EEG, conn, period=period)
    #compare_matrix_and_computed_psd(path = matrix_path, values = eeg_healthy, values2 = eeg_very_healty, visualizing = True)


if __name__ == "__main__":
    main()

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
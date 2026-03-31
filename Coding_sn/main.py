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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_plots", action="store_true", help="Show plots on screen")
    parser.add_argument("--supth", action="store_true", help="Switch to supra threshold tms stimulation")
    parser.add_argument("--pot", action="store_true", help="Switch to tms potentiation effect")
    parser.add_argument("--tms", action="store_true", help="Applay TMS")
    parser.add_argument("--conn_sim", action="store_true", help="Use 86 regions connectivity")
    parser.add_argument("--no_sim", action="store_true", help="Search for simulated data")
    parser.add_argument("--no_stim_sim", action="store_true", help="Perform unstimulated simuation")
    parser.add_argument("--save_plots", action="store_true", help="Save plots to disk")
    parser.add_argument("--save_folder", type=str, help="Folder to save figures and data")
    parser.add_argument("--save_raw_data", action="store_true", help="Saving raw eeg and tavg data")
    parser.add_argument("--struct_conn", type=str, help="Path to structural connectivity file")
    parser.add_argument("--EEG", action="store_true", help="Perform eeg")
    parser.add_argument("--stim", action="store_true", help="Perform TMS stimulation")
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
    parser.add_argument("--rnd", type=int, help="Simulation random seed")
    parser.add_argument("--tms_freq", type=float, help="Frequqncy of stimulation (choose between 0.5, 10, 20 Hz)")
    parser.add_argument(
    "--tau_array", 
    type=int, 
    nargs=3, 
    help="Lista degli stadi Tau (es: --tau_array 0 0 0)"
    )
    parser.add_argument(
    "--abeta_array", 
    type=int, 
    nargs=2, 
    help="Lista degli stadi Abeta (es: --abeta_array 0 0)"
    )
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
    gv.save_raw_data = args.save_raw_data
    gv.eeg = args.EEG
    gv.psd = args.PSD
    gv.fc = args.FC
    gv.brain_activity = args.brain_act
    gv.skin_and_sensors = args.skin_and_sens
    gv.sim_vs_real = args.sim_vs_real
    gv.connectivity_simnibs = args.conn_sim
    gv.stim = args.stim
    gv.no_stim_simulation = args.no_stim_sim
    gv.tms_param_variation = args.tms
    gv.supra_threshold =args.supth
    gv.potent_effect = args.pot

    if args.tms_freq:
        if (args.tms_freq == 0.5) | (args.tms_freq == 10.0) | (args.tms_freq == 20.0):
            gv.tms_freq = args.tms_freq
        else:
            print('invalid value of tms_freq, default value stimulus set')

    if args.g:
        gv.coup = args.g

    if args.rnd:
        gv.random_seed = args.rnd

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
        if args.tau_array:
            gv.tau_stage = args.tau_array
        else:
            gv.tau_stage = 0
    
    if args.abeta:
        gv.abeta_stage = args.abeta
    else:
        if args.abeta_array:
            gv.abeta_stage = args.abeta_array
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

    gv.connectivity_simnibs = True
    np.random.seed(gv.random_seed)
    if gv.no_stim_simulation == True:
        if args.no_sim:
            sim_id = f"tau{args.tau}_abeta{args.abeta}_a{args.a:.4f}_b{args.b:.4f}_g{args.g:.4f}_v{args.vel:.2f}_n{args.noise:.4f}_simtime{args.st}"
            # 2. data loading
            ttavg, tavg, teeg, eeg = load_simulation_results(sim_id)
            ttavg, tavg = preprocess(ttavg, tavg)
            teeg, eeg = preprocess(teeg, eeg)
        
        if not args.no_sim:
            if gv.connectivity_simnibs == True:
                #print('86 regioni')
                #print(gv.connectivity_simnibs)
                ttavg, tavg, teeg, eeg = new_parcellation_simulate(g=gv.coup, velocity=gv.vel, noise=gv.noi, a=gv.aa, b=gv.bb, factor=gv.factor, abeta_stage=gv.abeta_stage, tau_stage=gv.tau_stage)#simulate(g=gv.coup, velocity=gv.vel, noise=gv.noi, a=gv.aa, b=gv.bb, tau_stage=gv.tau_stage, abeta_stage=gv.abeta_stage)
            else:
                ttavg, tavg, teeg, eeg = simulate(g=gv.coup, velocity=gv.vel, noise=gv.noi, a=gv.aa, b=gv.bb,tau_stage=gv.tau_stage, abeta_stage=gv.abeta_stage, factor=gv.factor, save_raw = gv.save_raw_data)#simulate(g=gv.coup, velocity=gv.vel, noise=gv.noi, a=gv.aa, b=gv.bb, tau_stage=gv.tau_stage, abeta_stage=gv.abeta_stage)
            #ttavg, tavg, teeg, eeg = simulate_different_combination_parameters(g=gv.coup, velocity=gv.vel, noise=gv.noi, a=gv.aa, b=gv.bb,tau_stage=gv.tau_stage, abeta_stage=gv.abeta_stage, factor=gv.factor)#simulate(g=gv.coup, velocity=gv.vel, noise=gv.noi, a=gv.aa, b=gv.bb, tau_stage=gv.tau_stage, abeta_stage=gv.abeta_stage)
            #ttavg, tavg, teeg, eeg = Simulate()
            ttavg, tavg = preprocess(ttavg, tavg)
            teeg, eeg = preprocess(teeg, eeg)
        
        
        datas = {'data_healthy': {'tavg': tavg, 'eeg': eeg, 'label': 'Healthy', 'color': 'blue'}}
        
        
        if gv.skin_and_sensors == True:
            skin_and_sensors()
                # Plot cortical regions colorate per valore
            #skin = surfaces.SkinAir.from_file(r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\sub-CLB00044_outer_skin.zip")
            #skin.configure()
            #cortex = surfaces.CorticalSurface.from_file(r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\sub-CLB00044_Cortex.zip")
            #cortex.configure()
            #x_cortex, y_cortex, z_cortex = cortex.vertices.T
            #plt.figure(figsize=(6, 6))
            #ax = plt.subplot(111, projection='3d')
            #conn = gv.get_conn()
            #x_conn, y_conn, z_conn = conn.centres.T[:, :]
            
            #region_values = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\stimulus_vector_TVB.npy")
        """
            region_values = np.array([2., 2., 3., 3., 1., 2., 3., 2., 2., 2., 2., 2., 2., 2., 1., 3., 2.,
        2., 2., 2., 2., 1., 3., 3., 2., 2., 3., 3., 2., 2., 2., 1., 2., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 2.,
        3., 3., 1., 2., 3., 2., 2., 2., 2., 2., 2., 2., 1., 3., 2., 2., 2.,
        2., 3., 2., 3., 3., 2., 2., 3., 3., 2., 2., 2., 1., 2., 1., 1., 1.,
        1.])
            region_values_76 = np.zeros((76,1))

            for i in range(len(region_values_76)):
                if i < 20:
                    region_values_76[i] = 1
                if 20 <= i < 40:
                    region_values_76[i] = 2
                if i >= 40:
                    region_values_76[i] = 3
            region_values = region_values_76
            if region_values is not None:
                sc = ax.scatter(x_conn, y_conn, z_conn, c=region_values, cmap='hot', s=50, zorder=5)
                plt.colorbar(sc, ax=ax, shrink=0.5, label='Valore')
            else:
                ax.plot(x_conn, y_conn, z_conn, 'ko')
            
            ax.plot(x_conn, y_conn, z_conn, 'ko')
            #ax.plot_trisurf(x_cortex, y_cortex, z_cortex, triangles=cortex.triangles, alpha=0.1, edgecolor='k')

                # Plot corteccia colorata per regione
            
            if region_values is not None:
                # Per ogni vertice, trova la regione più vicina
                tree = KDTree(conn.centres)
                _, nearest_region = tree.query(cortex.vertices)

                # Mappa i valori delle regioni sui vertici
                vertex_values = region_values[nearest_region]
                
                # Calcola il colore medio per ogni triangolo
                tri_values = vertex_values[cortex.triangles].mean(axis=1)
                norm = plt.Normalize(vmin=region_values.min(), vmax=region_values.max())
                cmap = plt.cm.hot
                face_colors = cmap(norm(tri_values))
                # Costruisci i triangoli come lista di coordinate
                verts = cortex.vertices[cortex.triangles]  # shape (M, 3, 3)

                poly = Poly3DCollection(verts, alpha=0.8, edgecolor='none')
                poly.set_facecolor(face_colors)
                ax.add_collection3d(poly)

                # Aggiorna i limiti degli assi manualmente
                ax.set_xlim(x_cortex.min(), x_cortex.max())
                ax.set_ylim(y_cortex.min(), y_cortex.max())
                ax.set_zlim(z_cortex.min(), z_cortex.max())
            else:
                ax.plot_trisurf(x_cortex, y_cortex, z_cortex, triangles=cortex.triangles,
                                alpha=0.1, edgecolor='k')
            
            #x, y, z = skin.vertices.T
            #ax.plot_trisurf(x, y, z, triangles=skin.triangles, alpha=0.1, edgecolor='k')
            plt.show()
        """

        if gv.brain_activity == True:
            brain_activity(eeg=eeg[:,0,:,0], tavg=tavg[:,0,:,0])

        if gv.eeg == True:
            eeg_plot(datas = datas, times = teeg)

        """il controllo sulla label della FC va migliorato, vorrei si potesse inserire da terminale, magari facendo si che alla tupla di parametri richiesti 
        venga associata una certa dicita (e.g. malato braak stage 2 - beta stage 1)"""
        """
        if gv.fc == True:
            functional_connectivity_band(eeg, 'EEG alpha band (8-14)Hz', lowcut=8, highcut=14)
            functional_connectivity_band(eeg, 'EEG theta band (4-8)Hz', lowcut=4, highcut=8)
            functional_connectivity_band(eeg, 'EEG delta band (0.5-4)Hz', lowcut=0.5, highcut=4)
            functional_connectivity_band(tavg, 'TAVG alpha band (8-14)Hz', lowcut=8, highcut=14)
            functional_connectivity_band(tavg, 'TAVG theta band (4-8)Hz', lowcut=4, highcut=8)
            functional_connectivity_band(tavg, 'TAVG delta band (0.5-4)Hz', lowcut=0.5, highcut=4)
            functional_connectivity(eeg, 'EEG')
            functional_connectivity(tavg, 'TAVG')
        """
        if gv.fc == True:
            #functional_connectivity(tavg, label='tavg')
            
            #functional_connectivity_lorenzo(tavg)
            #functional_connectivity_band(tavg, label='TAVG', lowcut=0.5, highcut=45)
            functional_connectivity_band(tavg, label='TAVG delta', lowcut=0.5, highcut=4)
            #functional_connectivity_band(tavg, label='TAVG theta', lowcut=4, highcut=8)
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

        if gv.real_data == True:
            analyze_regular_data(np.load(gv.HC_FC_data))
            analyze_reduced_data(np.load(gv.HC_FC_data))
        
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
    
    if gv.stim == True:

        def ration_P30_N100(teeg, eeg, stimulus_onset, preprocess_window=1000):
            mask_baseline = teeg < (stimulus_onset-100)
            print(teeg[mask_baseline])
            mask_n15_to_p60 = (teeg >= stimulus_onset) & (teeg <= stimulus_onset + 50)
            print(teeg[mask_n15_to_p60])
            mask_n100 = (teeg > stimulus_onset + 70) & (teeg <= stimulus_onset + 150)
            print(teeg[mask_n100])
            assert(stimulus_onset >= preprocess_window+1000)
            eeg_mask_baseline = eeg[mask_baseline]
            eeg_mask_n15_to_p60 = eeg[mask_n15_to_p60]
            eeg_mask_n100 = eeg[mask_n100]
            eeg_baseline = np.mean(eeg_mask_baseline)
            n15_to_p60_amplitude = np.max(np.abs(eeg_mask_n15_to_p60-eeg_baseline))
            n100_amplitude = np.max(np.abs(eeg_mask_n100-eeg_baseline))
            print('baseline: ', eeg_baseline)
            print('massima ampiezza primo picco: ', n15_to_p60_amplitude)
            print('massima ampiezza secondo picco: ', n100_amplitude)
            print('rapporto fra picco N100 e picco N15toP60: ', n100_amplitude/n15_to_p60_amplitude, '; vorremo qualcosa compreso fra ', 7.94/2.44, ' e ', 5.97/4.50)

        def actual_weights(ttavg, tavg, stimulus_onset):
            mask_baseline = ttavg < (stimulus_onset-100)
            tavg_baseline  = tavg[mask_baseline]
            amplitude_baseline = np.abs(np.max(tavg_baseline)- np.min(tavg_baseline))
            mask_TMS = ttavg > stimulus_onset
            tavg_TMS  = tavg[mask_TMS]
            amplitude_TMS = np.abs(np.max(tavg_TMS)- np.min(tavg_TMS))
            return amplitude_baseline, amplitude_TMS



        scale_fact = 38#10000 #con 0.63 otteniamo un rapporto 2picco/1picco = 3.23 #per ristabilire valore assoluto in output da simnibs sarebbe 0.366343
        start_time = 2500

        weights_vector = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\stimulus_vector_TVB.npy")#np.zeros(86)#
        weights_vector = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\simnibs_simulation_M1_stimulus_vector_TVB.npy")
        weights_vector = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\prova_protocollo_1_stimulus_vector_TVB.npy")
        #print(weights_vector)
        
        train_05Hz = make_train(np.arange(0,86), scale_fact*weights_vector, onset=start_time, T=2000.0, tau=1.0)
        train_10Hz = make_train(np.arange(0,86), scale_fact*weights_vector, onset=start_time, T=100.0, tau=1.0)
        train_20Hz = make_train(np.arange(0,86), scale_fact*weights_vector, onset=start_time, T=50.0, tau=1.0)
        
        """
        train1 = make_train([35,73], 5.0/stim_weight, onset=start_time , T=1000.0, tau=40.0) #500
        train2 = make_train([36,74], 4.5/stim_weight, onset=start_time + 10 * scale_fact, T=1000, tau=50.0) #600
        train3 = make_train([34,72], 4.0/stim_weight, onset=start_time + 35 * scale_fact, T=1000, tau=80.0) #850
        train4 = make_train([32,70], 4.0/stim_weight, onset=start_time + 40 * scale_fact, T=1000, tau=80.0) #900
        train5 = make_train([24,62], 4.0/stim_weight, onset=start_time + 50 * scale_fact, T=1000, tau=80.0) #1000
        stimulus = MultiStimuliRegion(train1, train2,train3, train4, train5)
        """
        
        stimulus = MultiStimuliRegion(train_05Hz)
        stimulus.configure_space()
        time = np.r_[1e3:3e3:15.0]
        stimulus.configure_time(time)
        pattern = stimulus()

        
        sim_time = 4000
        display_time = sim_time ###########################################

        ttavg_stim, tavg_healthy_stim, teeg_stim, eeg_healthy_stim = new_parcellation_simulate(g=gv.coup, velocity=gv.vel, noise=gv.noi, a=gv.aa, b=gv.bb, sim_time=sim_time, stimulus=stimulus, tau_stage=gv.tau_stage, abeta_stage=gv.abeta_stage)#simulate_with_stimulus(stimulus = stimulus, sim_time = sim_time) #Less simulated time needed
        ttavg_stim = ttavg_stim[1000:]
        tavg_healthy_stim = tavg_healthy_stim[1000:, :, :, :]
        teeg_stim, eeg_healthy_stim = preprocess(teeg_stim, eeg_healthy_stim)

        datas_stim = {'data_healthy': {'tavg': tavg_healthy_stim, 'eeg': eeg_healthy_stim, 'label': 'Healthy', 'color': 'blue'}}

        all_eeg_channels = np.arange(0,63)
        near_m1_channles_Pz_CPz = [3,17]#[30,32,33,8,9]
        channel_Cz = [0]#[21,31]
        chlist = channel_Cz

        mon_EEG = gv.get_eeg_monitor()

        for key, value in datas_stim.items():
            tavg_stim = value['tavg']
            eeg_stim = value['eeg']
            label = value['label']
            color = value['color']

            #Plot EEG
            plt.figure(figsize=(10,12))
            plt.plot(teeg_stim[:display_time], eeg_stim[:display_time, 0, chlist, 0] + np.r_[:(len(chlist))], color = color)
            plt.yticks(np.r_[:len(chlist)], mon_EEG.sensors.labels[chlist], fontsize = 12)
            plt.title("EEG sensors," + label, fontsize = 18)
            plt.xlabel('time (ms)', fontsize = 16)
            if gv.save_plots == True:
                plt.savefig(r"C:\Users\User\OneDrive - University of Pisa\Desktop\Immagini da far vedere a Lorenzo_20_03\Stimulation_channels_Pz_CPz\20_sensors.png")
            if gv.show_plots == True:
                plt.show()
            plt.close()

        marker = False
        if marker == True:
            tavg_list = [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
            tavg_list = [22, 71]

            tavg_channel_names = np.array(['L.BSTS', 'L.CACG', 'L.CMFG', 'L.CU', 'L.EC', 'L.FG', 'L.IPG', 'L.ITG', 'L.ICG',
                                'L.LOG', 'L.LOFG', 'L.LG', 'L.MOFG', 'L.MTG', 'L.PHIG', 'L.PaCG', 'L.POP', 'L.POR',
                                'L.PTR', 'L.PCAL', 'L.PoCG', 'L.PCG', 'L.PrCG', 'L.PCU', 'L.RACG', 'L.RMFG',
                                'L.SFG', 'L.SPG', 'L.STG', 'L.SMG', 'L.FP', 'L.TP', 'L.TTG', 'L.IN', 'L.CER',
                                'L.TH', 'L.CA', 'L.PU', 'L.PA', 'L.HI', 'L.AM', 'L.AC', 'R.TH', 'R.CA', 'R.PU',
                                'R.PA', 'R.HI', 'R.AM', 'R.AC', 'R.BSTS', 'R.CACG', 'R.CMFG', 'R.CU', 'R.EC', 'R.FG',
                                'R.IPG', 'R.ITG', 'R.ICG', 'R.LOG', 'R.LOFG', 'R.LG', 'R.MOFG', 'R.MTG', 'R.PHIG',
                                'R.PaCG', 'R.POP', 'R.POR', 'R.PTR', 'R.PCAL', 'R.PoCG', 'R.PCG', 'R.PrCG', 'R.PCU',
                                'R.RACG', 'R.RMFG', 'R.SFG', 'R.SPG', 'R.STG', 'R.SMG', 'R.FP', 'R.TP', 'R.TTG',
                                'R.IN', 'R.CER', 'L.DMR', 'R.DMR']) #L.DMR e R.DMR sono nomi inventati che ho dato alle dummy regions
            plt.figure(figsize=(10,12))
            plt.plot(ttavg_stim[:display_time], tavg_stim[:display_time, 0, tavg_list, 0] + np.r_[:(len(tavg_list))], color = color)
            plt.yticks(np.r_[:len(tavg_list)], tavg_channel_names[tavg_list], fontsize = 12)
            plt.title("EEG sensors," + label, fontsize = 18)
            plt.xlabel('time (ms)', fontsize = 16)
            if gv.save_plots == True:
                plt.savefig(r"C:\Users\User\OneDrive - University of Pisa\Desktop\Immagini da far vedere a Lorenzo_20_03\Stimulation_channels_Pz_CPz\20_sensors.png")
            if gv.show_plots == True:
                plt.show()
            plt.close()

        
        amplitude_TMS = []
        amplitude_baseline = []
        for i in range(len(tavg_stim[0, 0, :, 0])):
            amp_b, amp_TMS = actual_weights(ttavg=ttavg_stim, tavg=tavg_stim[:, 0, i, 0], stimulus_onset=start_time)
            amplitude_baseline.append(amp_b)
            amplitude_TMS.append(amp_TMS)
        
        amplitude_baseline = np.array(amplitude_baseline)
        amplitude_TMS = np.array(amplitude_TMS)
        np.save("amplitude_base_stage_11_05_Hz_38_stim.npy", amplitude_baseline)
        np.save("amplitude_TMS_stage_11_05_Hz_38_stim.npy", amplitude_TMS)
        
            
        ################################## USEFUL ONLY WHEN STIMULUS IS APPLIED ###################################################
        
        #1500 is stimulus onset 
        mask1 = 0
        mask2 = 1000
        #chlist_occipital = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]#[30,32,33,8,9]
        ration_P30_N100(teeg=teeg_stim, eeg=eeg_stim[:,0,0,0], stimulus_onset=start_time)
        stimulus_visualization(visualizing=True, stimulus_onset=start_time, datas_stim=datas_stim, teeg_stim=teeg_stim, mask1=mask1, mask2=mask2, chlist=chlist)

if __name__ == "__main__":
    main()
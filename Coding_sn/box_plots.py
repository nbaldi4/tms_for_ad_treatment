import pandas as pd
from pathlib import Path
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import global_variable_creation as gv
import sns
import seaborn

from data_analysis_utils import real_vs_stages_PSD, real_vs_stages_FC, to_common_indices, macro_fc
from additional_utils import extract_common_names
from sklearn.feature_selection import mutual_info_regression
from specparam import SpectralGroupModel # Usiamo Group per matrici [n_regions, n_freqs]

import re
import pandas as pd
import argparse
from pathlib import Path

def boxplots(data_bands, ylabel, xlabel, band_names, save_name, values_to_take, save_folder):
    for band in range(len(band_names)):
        
        n = data_bands.shape[2]
        k = n - values_to_take 
        top_idx = np.argpartition(data_bands, -values_to_take, axis=2)[:, :, -values_to_take:]
        data_bands_top_vals = np.take_along_axis(data_bands, top_idx, axis=2)
        
        data = data_bands_top_vals[:, band, :]
        
        plt.figure(figsize=(5,4))
        plt.boxplot(data.T)
        
        plt.title(band_names[band] + " band")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        # griglia solo asse y
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        
        # yticks personalizzati (esempio automatico)
        ymin = data.min() - (data.max() - data.min())/10
        ymax = data.max() + (data.max() - data.min())/10
        # if non specified matplotlib set ticks of boxplots starting from 1
        positions = np.arange(len(data)) + 1
        plt.xticks(positions, np.arange(len(data)))
        plt.yticks(np.linspace(ymin, ymax, 10))
        #plt.xticks(np.linspace(0, len(data_bands_top_vals[:,0,0]), len(data_bands_top_vals[:,0,0])))
        
        plt.tight_layout()
        #plt.show()
        folder = os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop", save_folder)
        os.makedirs(folder, exist_ok=True)
        plt.savefig(os.path.join(folder,save_name + str(band)))
        plt.close()
        print(band_names[band] + " band", xlabel)
        print(data.shape)
        for i in range(6):
            print(np.median(data[i, :]))

def patients_boxplots(ylabel, band_names):
    
    root_path = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data"
    mci_data = np.load(os.path.join(root_path, r"Patients_highlights\mci_summary_stats_ref_freqs.npz"), allow_pickle=True)
    mci_fc_abs_mean = mci_data['mci_fc_abs_mean']
    print(mci_fc_abs_mean[:5,:])
    mci_psd_rel_pow = mci_data['mci_psd_rel_pow']
    mci_csf_values = mci_data['mci_csf']

    for band in range(len(band_names)):
        data = mci_fc_abs_mean[:,band]
        plt.figure(figsize=(5,4))
        plt.boxplot(data)
        
        plt.title(band_names[band] + " band")
        plt.ylabel(ylabel)
        
        # griglia solo asse y
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        
        # yticks personalizzati (esempio automatico)
        ymin = data.min() - (data.max() - data.min())/10
        ymax = data.max() + (data.max() - data.min())/10
        # if non specified matplotlib set ticks of boxplots starting from 1
        plt.yticks(np.linspace(ymin, ymax, 10))
        #plt.xticks(np.linspace(0, len(data_bands_top_vals[:,0,0]), len(data_bands_top_vals[:,0,0])))
        
        plt.tight_layout()
        #plt.show()
        folder = os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop", 'patients_boxplots')
        os.makedirs(folder, exist_ok=True)
        plt.savefig(os.path.join(folder,'mci patients ' + str(band)))
        plt.close()

    psd_band_names = ['Delta/Broad', 'Theta/Broad', 'Alpha/Broad', 'Beta/Broad']

    for band in range(len(psd_band_names)):
        data_psd = mci_psd_rel_pow[:,band]
        plt.figure(figsize=(5,4))
        plt.boxplot(data_psd)
        
        plt.title("PSD " + psd_band_names[band])
        plt.ylabel("relative PSD")
        
        # griglia solo asse y
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        
        # yticks personalizzati (esempio automatico)
        ymin = data_psd.min() - (data_psd.max() - data_psd.min())/10
        ymax = data_psd.max() + (data_psd.max() - data_psd.min())/10
        # if non specified matplotlib set ticks of boxplots starting from 1
        plt.yticks(np.linspace(ymin, ymax, 10))
        #plt.xticks(np.linspace(0, len(data_bands_top_vals[:,0,0]), len(data_bands_top_vals[:,0,0])))
        
        plt.tight_layout()
        #plt.show()
        folder = os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop", 'patients_boxplots')
        os.makedirs(folder, exist_ok=True)
        plt.savefig(os.path.join(folder,'mci patients' + str(band) + 'PSD'))
        plt.close()

    data2 = np.zeros((len(mci_fc_abs_mean[:,0]), len(mci_fc_abs_mean[0,:])-1))
    for band in range(len(band_names)-1):
        print("banda ", band_names[band+1])
        for i in range(len(mci_fc_abs_mean[:,band])):
            if i <= 5:
                print(mci_fc_abs_mean[i,:])
                data2[i, band] = (mci_fc_abs_mean[i,band+1]-mci_fc_abs_mean[i,0])/mci_fc_abs_mean[i,0]
                print("paziente ", i)
                print(data2[i, band])
                print(mci_fc_abs_mean[i, band+1])
                print(mci_fc_abs_mean[i, 0])
        plt.figure(figsize=(5,4))
        plt.boxplot(data2)
        
        plt.title(band_names[band+1] + " band over broad")
        plt.ylabel(ylabel)
        
        # griglia solo asse y
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        
        # yticks personalizzati (esempio automatico)
        ymin = data2.min() - (data2.max() - data2.min())/10
        ymax = data2.max() + (data2.max() - data2.min())/10
        # if non specified matplotlib set ticks of boxplots starting from 1
        plt.yticks(np.linspace(ymin, ymax, 10))
        #plt.xticks(np.linspace(0, len(data_bands_top_vals[:,0,0]), len(data_bands_top_vals[:,0,0])))
        
        plt.tight_layout()
        #plt.show()
        folder = os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop", 'patients_boxplots')
        os.makedirs(folder, exist_ok=True)
        plt.savefig(os.path.join(folder,'mci patients over broad' + str(band+1)))
        plt.close()
    

def main(
stages_folder = "Rnd_seeds_M_FC_pears_tavg", seeds_number = 40, values_to_take=40, psd = False, fc = False
):
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--psd", action="store_true", help="psd box plot")
    parser.add_argument("--fc", action="store_true", help="fc box plot")
    parser.add_argument("--patients", action="store_true", help="patients box plot")
    parser.add_argument("--stages_folder", type=str, help="Folder to analyze")
    parser.add_argument("--save_folder", type=str, help="Folder where save data")
    parser.add_argument("--seeds_number", type=int, help="number of seeds being analyzed")
    parser.add_argument("--values_to_take", type=int, help="highest values to enter the boxplots")

    args = parser.parse_args()
    if args.stages_folder:
        stages_folder = args.stages_folder

    if args.save_folder:
        save_folder = args.save_folder
    else:
        save_folder = "prova_box_plots"
    
    if args.patients:
        patients = args.patients
    else:
        patients = False
    
    if args.fc:
        fc = args.fc
    else:
        fc = False
    
    if args.psd:
        psd = args.psd
    else:
        psd = False
    
    if args.values_to_take:
        values_to_take = args.values_to_take

    if args.seeds_number:
        seeds_number = args.seeds_number

    input_path_fc = os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Staging_highlights", stages_folder + '_stages_fc.npz')

    input_path_psd = os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Staging_highlights", stages_folder + '_stages_psd.npz')


    loaded_psd_data = np.load(input_path_psd, allow_pickle=True)
    loaded_fc_data = np.load(input_path_fc, allow_pickle=True)
    fc_from_file = loaded_fc_data['fc_matrices']
    psd_from_file = loaded_psd_data['psd_arrays']
    freqs_from_file = loaded_psd_data['psd_freqs']

    print(fc_from_file.shape)
    """
    for seed in range(seeds_number):
        print('###### abeta = 0 ###### tau = 1', seed, '######')
        print('broad band')
        print(np.mean(np.abs(fc_from_file[0, 1, 0, seed])))
        print('delta band')
        print(np.mean(np.abs(fc_from_file[0, 1, 1, seed])))
        print('theta band')
        print(np.mean(np.abs(fc_from_file[0, 1, 2, seed])))
        print('alpha band')
        print(np.mean(np.abs(fc_from_file[0, 1, 3, seed])))
        print('beta band')
        print(np.mean(np.abs(fc_from_file[0, 1, 4, seed])))
        print('###### abeta = 1 ###### tau = 0', seed, '######')
        print('broad band')
        print(np.mean(np.abs(fc_from_file[1, 0, 0, seed])))
        print('delta band')
        print(np.mean(np.abs(fc_from_file[1, 0, 1, seed])))
        print('theta band')
        print(np.mean(np.abs(fc_from_file[1, 0, 2, seed])))
        print('alpha band')
        print(np.mean(np.abs(fc_from_file[1, 0, 3, seed])))
        print('beta band')
        print(np.mean(np.abs(fc_from_file[1, 0, 4, seed])))
    """

    stages = 6
    bands = 5
    seeds = seeds_number
    psd_metrics = np.zeros((stages,stages,bands-1,seeds))

    for abeta in range(stages):
        for tau in range(stages):
            for seed in range(seeds):
                freqs = freqs_from_file[abeta,tau,seed]
                psd_broad = np.sum(psd_from_file[abeta,tau,seed][(freqs>= 0.5) & (freqs < 45)])  #broad
                psd_delta = np.sum(psd_from_file[abeta,tau,seed][(freqs>= 0.5) & (freqs < 4)])  #delta
                psd_theta = np.sum(psd_from_file[abeta,tau,seed][(freqs >= 4) & (freqs < 8)])   #theta
                psd_alpha = np.sum(psd_from_file[abeta,tau,seed][(freqs >= 8) & (freqs < 13)])  #alpha
                psd_beta = np.sum(psd_from_file[abeta,tau,seed][(freqs >= 13) & (freqs < 30)])  #beta
                
                psd_metrics[abeta,tau,0,seed] = psd_delta/psd_broad
                psd_metrics[abeta,tau,1,seed] = psd_theta/psd_broad
                psd_metrics[abeta,tau,2,seed] = psd_alpha/psd_broad
                psd_metrics[abeta,tau,3,seed] = psd_beta/psd_broad  

    #print(psd_metrics[0,0,0,:])
    """
    for i in range(20):
        print('abeta 0, tau 0, banda alpha, seed ',i)
        print(fc_from_file[0, 0, 3, i].shape, '; ', np.mean(np.abs(fc_from_file[0, 0, 3, i])))
    """

    fc_means_tau_abeta = np.zeros((stages, bands, seeds))
    for prot in range(stages):
        for band in range(bands):
            for seed in range(seeds):
                fc_means_tau_abeta[prot, band, seed] = np.mean(np.abs(fc_from_file[prot, prot, band, seed]))

    fc_means_tau_0 = np.zeros((stages, bands, seeds))
    for tau in range(stages):
        for band in range(bands):
            for seed in range(seeds):
                fc_means_tau_0[tau, band, seed] = np.mean(np.abs(fc_from_file[0, tau, band, seed]))
            
    fc_means_0_abeta = np.zeros((stages, bands, seeds))
    for abeta in range(stages):
        for band in range(bands):
            for seed in range(seeds):
                fc_means_0_abeta[abeta, band, seed] = np.mean(np.abs(fc_from_file[abeta, 0, band, seed]))

    psd_means_tau_abeta = np.zeros((stages, bands-1, seeds))
    for prot in range(stages):
        for band in range(bands-1):
            for seed in range(seeds):
                psd_means_tau_abeta[prot, band, seed] = psd_metrics[prot, prot, band, seed]

    psd_means_tau_0 = np.zeros((stages, bands-1, seeds))
    for tau in range(stages):
        for band in range(bands-1):
            for seed in range(seeds):
                psd_means_tau_0[tau, band, seed] = psd_metrics[0, tau, band, seed]
            
    psd_means_0_abeta = np.zeros((stages, bands-1, seeds))
    for abeta in range(stages):
        for band in range(bands-1):
            for seed in range(seeds):
                psd_means_0_abeta[abeta, band, seed] = psd_metrics[abeta, 0, band, seed]

    fc_means_tau_abeta_band_over_broad = np.zeros((stages,bands-1,seeds))
    for prot in range(stages):
        for seed in range(seeds):
            fc_means_tau_abeta_band_over_broad[prot, 0, seed] = (fc_means_tau_abeta[prot, 1, seed]-fc_means_tau_abeta[prot, 0, seed])/fc_means_tau_abeta[prot, 0, seed]
            fc_means_tau_abeta_band_over_broad[prot, 1, seed] = (fc_means_tau_abeta[prot, 2, seed]-fc_means_tau_abeta[prot, 0, seed])/fc_means_tau_abeta[prot, 0, seed]
            fc_means_tau_abeta_band_over_broad[prot, 2, seed] = (fc_means_tau_abeta[prot, 3, seed]-fc_means_tau_abeta[prot, 0, seed])/fc_means_tau_abeta[prot, 0, seed]
            fc_means_tau_abeta_band_over_broad[prot, 3, seed] = (fc_means_tau_abeta[prot, 4, seed]-fc_means_tau_abeta[prot, 0, seed])/fc_means_tau_abeta[prot, 0, seed]

    fc_band_names = ['Broad', 'Delta', 'Theta', 'Alpha', 'Beta']
    psd_band_names = ['Delta/Broad', 'Theta/Broad', 'Alpha/Broad', 'Beta/Broad']

    if fc:
        boxplots(fc_means_tau_abeta, xlabel="Diagonal (tau = abeta)", ylabel="FC regular abs mean", band_names=fc_band_names, save_name='fc_diagonal_', values_to_take=values_to_take, save_folder=save_folder)
        boxplots(fc_means_tau_0, xlabel="Tau (abeta = 0)", ylabel="FC regular abs mean", band_names=fc_band_names, save_name='fc_tau_', values_to_take=values_to_take, save_folder=save_folder)
        boxplots(fc_means_0_abeta, xlabel="Abeta (tau = 0)", ylabel="FC regular abs mean", band_names=fc_band_names, save_name='fc_abeta_', values_to_take=values_to_take, save_folder=save_folder)
        boxplots(fc_means_tau_abeta_band_over_broad, xlabel="Diagonal (tau = abeta)", ylabel="FC regular abs mean normalized", band_names=psd_band_names, save_name='fc_over_broad_diagonal', values_to_take=values_to_take, save_folder=save_folder)
    if psd:
        boxplots(psd_means_tau_abeta, xlabel="Diagonal (tau = abeta)", ylabel="Relative band power", band_names=psd_band_names, save_name='psd_diagonal_', values_to_take=values_to_take, save_folder=save_folder)
        boxplots(psd_means_tau_0, xlabel="Tau (abeta = 0)", ylabel="Relative band power", band_names=psd_band_names, save_name='psd_tau_', values_to_take=values_to_take, save_folder=save_folder)
        boxplots(psd_means_0_abeta, xlabel="Abeta (tau = 0)", ylabel="Relative band power", band_names=psd_band_names, save_name='psd_abeta_', values_to_take=values_to_take, save_folder=save_folder)
    if patients:
        patients_boxplots(ylabel="FC regular abs mean normalized", band_names=fc_band_names)



    #path_patient_203 = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\patient_203\PSD_EEG_patient_203.npz")

if __name__ == "__main__":
    main()

"""
print(loaded_psd_data.files)
print(freqs_from_file[3, 5, 5])

print(fc_from_file.shape)
print(psd_from_file.shape)
print(fc_from_file[0,0,0,0].shape)
print(psd_from_file[0,0,0].shape)

print('abeta 0, tau 4, banda theta, seed 3')
print(fc_from_file[0, 4, 2, 3].shape, '; ', np.mean(np.abs(fc_from_file[0, 4, 2, 3])))
print(psd_from_file[0, 4, 3].shape, '; ', np.max(psd_from_file[0, 4, 3]))
print(freqs_from_file[0, 4, 3].shape)
print('abeta 1, tau 5, banda alpha, seed 12')
print(fc_from_file[1, 5, 3, 12].shape, '; ', np.mean(np.abs(fc_from_file[1, 5, 3, 12])))
print(psd_from_file[1, 5, 12].shape, '; ', np.max(psd_from_file[1, 5, 12]))
print(freqs_from_file[1, 5, 12].shape)
print('abeta 2, tau 1, banda delta, seed 1')
print(fc_from_file[2, 1, 1, 1].shape, '; ', np.mean(np.abs(fc_from_file[2, 1, 1, 1])))
print(psd_from_file[2, 1, 1].shape, '; ', np.max(psd_from_file[2, 1, 1]))
print(freqs_from_file[2, 1, 1].shape)

print(path_patient_203.files)
patient_203_freqs = path_patient_203['freqs']
print(patient_203_freqs.shape)
print(patient_203_freqs)

is_identical = all(np.array_equal(a, b) for a, b in zip(freqs_from_file[3, 5, 5].flat, patient_203_freqs.flat))
if is_identical:
    print("✅ Identical freqs")
else:
    print("❌ freqs differiscono nel contenuto")

plt.figure(figsize=(10, 6))
plt.plot(freqs_from_file[0, 5, 2], psd_from_file[1, 5, 12], linewidth=3)
#plt.ylim(0, 1.01)
plt.xlim(0, 45)
plt.xlabel('Frequency [Hz]', fontsize=16)
plt.ylabel('PSD', fontsize=16)
plt.grid(True)
plt.show()
"""
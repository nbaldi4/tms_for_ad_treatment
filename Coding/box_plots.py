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
from pathlib import Path
stages_folder = 'Random_seeds_16_02_M'
output_path_fc = os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Staging_highlights", stages_folder + "_stages_fc.npz")

output_path_psd = os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Staging_highlights", stages_folder + "_stages_psd.npz")

loaded_psd_data = np.load(output_path_psd, allow_pickle=True)
loaded_fc_data = np.load(output_path_fc, allow_pickle=True)
fc_from_file = loaded_fc_data['fc_matrices']
psd_from_file = loaded_psd_data['psd_arrays']
freqs_from_file = loaded_psd_data['psd_freqs']

for seed in range(20):
    print('broad band')
    print(np.mean(np.abs(fc_from_file[0, 0, 0, seed])))
    print('delta band')
    print(np.mean(np.abs(fc_from_file[0, 0, 1, seed])))
    print('theta band')
    print(np.mean(np.abs(fc_from_file[0, 0, 2, seed])))
    print('alpha band')
    print(np.mean(np.abs(fc_from_file[0, 0, 3, seed])))
    print('beta band')
    print(np.mean(np.abs(fc_from_file[0, 0, 4, seed])))

psd_metrics = np.zeros((6,6,4,20))

for abeta in range(6):
    for tau in range(6):
        for seed in range(20):
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
fc_means_tau_abeta = np.zeros((6, 5, 20))
for prot in range(6):
    for band in range(5):
        for seed in range(20):
            fc_means_tau_abeta[prot, band, seed] = np.mean(np.abs(fc_from_file[prot, prot, band, seed]))

fc_means_tau_0 = np.zeros((6, 5, 20))
for tau in range(6):
    for band in range(5):
        for seed in range(20):
            fc_means_tau_0[tau, band, seed] = np.mean(np.abs(fc_from_file[0, tau, band, seed]))
        
fc_means_0_abeta = np.zeros((6, 5, 20))
for abeta in range(6):
    for band in range(5):
        for seed in range(20):
            fc_means_0_abeta[abeta, band, seed] = np.mean(np.abs(fc_from_file[abeta, 0, band, seed]))

psd_means_tau_abeta = np.zeros((6, 4, 20))
for prot in range(6):
    for band in range(4):
        for seed in range(20):
            psd_means_tau_abeta[prot, band, seed] = psd_metrics[prot, prot, band, seed]

psd_means_tau_0 = np.zeros((6, 4, 20))
for tau in range(6):
    for band in range(4):
        for seed in range(20):
            psd_means_tau_0[tau, band, seed] = psd_metrics[0, tau, band, seed]
        
psd_means_0_abeta = np.zeros((6, 4, 20))
for abeta in range(6):
    for band in range(4):
        for seed in range(20):
            psd_means_0_abeta[abeta, band, seed] = psd_metrics[abeta, 0, band, seed]

fc_means_tau_abeta_band_over_broad = np.zeros((6,4,20))
for prot in range(6):
    for seed in range(20):
        fc_means_tau_abeta_band_over_broad[prot, 0, seed] = (fc_means_tau_abeta[prot, 1, seed]-fc_means_tau_abeta[prot, 0, seed])/fc_means_tau_abeta[prot, 0, seed]
        fc_means_tau_abeta_band_over_broad[prot, 1, seed] = (fc_means_tau_abeta[prot, 2, seed]-fc_means_tau_abeta[prot, 0, seed])/fc_means_tau_abeta[prot, 0, seed]
        fc_means_tau_abeta_band_over_broad[prot, 2, seed] = (fc_means_tau_abeta[prot, 3, seed]-fc_means_tau_abeta[prot, 0, seed])/fc_means_tau_abeta[prot, 0, seed]
        fc_means_tau_abeta_band_over_broad[prot, 3, seed] = (fc_means_tau_abeta[prot, 4, seed]-fc_means_tau_abeta[prot, 0, seed])/fc_means_tau_abeta[prot, 0, seed]

fc_band_names = ['Broad', 'Delta', 'Theta', 'Alpha', 'Beta']
psd_band_names = ['Delta/Broad', 'Theta/Broad', 'Alpha/Broad', 'Beta/Broad']

def boxplots(data_bands, ylabel, xlabel, band_names, save_name):
    for band in range(len(band_names)):
        
        data = data_bands[:, band, :]
        
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
        plt.yticks(np.linspace(ymin, ymax, 10))
        
        plt.tight_layout()
        plt.show()
        #plt.savefig(os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop\Immagini da far vedere a Lorenzo",save_name + str(band)))
        #plt.close()

#boxplots(fc_means_tau_abeta, xlabel="Diagonal (tau = abeta)", ylabel="FC regular abs mean", band_names=fc_band_names, save_name='fc_diagonal_')
#boxplots(fc_means_tau_0, xlabel="Tau (abeta = 0)", ylabel="FC regular abs mean", band_names=fc_band_names, save_name='fc_tau_')
#boxplots(fc_means_0_abeta, xlabel="Abtea (tau = 0)", ylabel="FC regular abs mean", band_names=fc_band_names, save_name='fc_abeta_')
#boxplots(fc_means_tau_abeta_band_over_broad, xlabel="Diagonal (tau = abeta)", ylabel="FC regular abs mean normalized", band_names=psd_band_names, save_name='fc_over_broad_diagonal')
#boxplots(psd_means_tau_abeta, xlabel="Diagonal (tau = abeta)", ylabel="Relative band power", band_names=psd_band_names, save_name='psd_diagonal_')
#boxplots(psd_means_tau_0, xlabel="Tau (abeta = 0)", ylabel="Relative band power", band_names=psd_band_names, save_name='psd_tau_')
#boxplots(psd_means_0_abeta, xlabel="Abtea (tau = 0)", ylabel="Relative band power", band_names=psd_band_names, save_name='psd_abeta_')



#path_patient_203 = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\patient_203\PSD_EEG_patient_203.npz")
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
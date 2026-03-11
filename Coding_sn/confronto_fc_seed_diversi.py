import pandas as pd
from pathlib import Path
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import global_variable_creation as gv
import sns

from data_analysis_utils import real_vs_stages_PSD, real_vs_stages_FC, to_common_indices, macro_fc
from additional_utils import extract_common_names
from sklearn.feature_selection import mutual_info_regression
from specparam import SpectralGroupModel # Usiamo Group per matrici [n_regions, n_freqs]


# Rimuove il limite di larghezza della colonna (None = nessun limite)
pd.set_option('display.max_colwidth', None)
file_path_1_EEG = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\tau0_abeta0_a0.0728_b0.0629_g0.1000_v10.37_n0.1448_simtime20000_rnd10\FC_EEG_tau_0_abeta_0_0.5-45_Hz_rnd10.npz"
file_path_2_EEG = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\tau0_abeta0_a0.0728_b0.0629_g0.1000_v10.37_n0.1448_simtime20000_rs0\FC_EEG_tau_0_abeta_0_0.5-45_Hz.npz"
file_path_1_TAVG = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\tau0_abeta0_a0.0728_b0.0629_g0.1000_v10.37_n0.1448_simtime20000_rnd10\FC_TAVG_tau_0_abeta_0_0.5-45_Hz_rnd10.npz"
file_path_2_TAVG = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\tau0_abeta0_a0.0728_b0.0629_g0.1000_v10.37_n0.1448_simtime20000_rs0\FC_TAVG_tau_0_abeta_0_0.5-45_Hz.npz"
fc_1_EEG = np.load(file_path_1_EEG)
fc_1_TAVG = np.load(file_path_1_TAVG)
fc_2_EEG = np.load(file_path_2_EEG)
fc_2_TAVG = np.load(file_path_2_TAVG)
fc_matrix_1_EEG = fc_1_EEG['normal_size']
fc_matrix_1_TAVG = fc_1_TAVG['normal_size']
fc_matrix_2_EEG = fc_2_EEG['normal_size']
fc_matrix_2_TAVG = fc_2_TAVG['normal_size']
sub_corr_viz = 2 * (fc_matrix_1_EEG - np.min(fc_matrix_1_EEG)) / (np.max(fc_matrix_1_EEG) - np.min(fc_matrix_1_EEG)) - 1
# Plot corr
fig = plt.figure(figsize=(12, 8)) 
ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
im = ax.imshow(sub_corr_viz, cmap='bwr', vmin=-1, vmax=1, aspect='equal')
abs_corr_mean = np.mean(np.abs(fc_matrix_1_EEG))
corr_mean = np.mean(fc_matrix_1_EEG)
title_text = f"Reduced size simulated FC; FC reduced mean = {corr_mean:.4f}, ({abs_corr_mean:.4f})"
ax.set_title(title_text, fontsize=15)
cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
plt.colorbar(im, cax=cax)
sub_corr_viz_2 = 2 * (fc_matrix_2_EEG - np.min(fc_matrix_2_EEG)) / (np.max(fc_matrix_2_EEG) - np.min(fc_matrix_2_EEG)) - 1
# Plot corr
fig = plt.figure(figsize=(12, 8)) 
ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
im = ax.imshow(sub_corr_viz_2, cmap='bwr', vmin=-1, vmax=1, aspect='equal')
abs_corr_mean_2 = np.mean(np.abs(fc_matrix_2_EEG))
corr_mean_2 = np.mean(fc_matrix_2_EEG)
title_text = f"Reduced size simulated FC; FC reduced mean = {corr_mean_2:.4f}, ({abs_corr_mean_2:.4f})"
ax.set_title(title_text, fontsize=15)
cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
plt.colorbar(im, cax=cax)
plt.show()

print(fc_matrix_1_EEG.shape, fc_matrix_2_EEG.shape)
print(np.isnan(fc_matrix_1_EEG).any(), np.isnan(fc_matrix_2_EEG).any())
print(fc_matrix_1_TAVG.shape, fc_matrix_2_TAVG.shape)
print(np.isnan(fc_matrix_1_TAVG).any(), np.isnan(fc_matrix_2_TAVG).any())
# prendi solo triangolo superiore (senza diagonale)
iu = np.triu_indices_from(fc_matrix_1_EEG, k=1)
r_EEG = np.corrcoef(fc_matrix_1_EEG[iu], fc_matrix_2_EEG[iu])[0, 1]
print("FC–FC correlation (EEG):", r_EEG)
iu = np.triu_indices_from(fc_matrix_1_TAVG, k=1)
r_TAVG = np.corrcoef(fc_matrix_1_TAVG[iu], fc_matrix_2_TAVG[iu])[0, 1]
print("FC–FC correlation (TAVG):", r_TAVG)

import pandas as pd
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import global_variable_creation as gv

from data_analysis_utils import real_vs_stages_PSD, real_vs_stages_FC, to_common_indices, macro_fc


# Rimuove il limite di larghezza della colonna (None = nessun limite)
pd.set_option('display.max_colwidth', None)

def collect_files(root_folder_path, files_name):
    """
    Raccoglie tutti i file che iniziano con 'FC_EEG alpha band' in tutte le sottocartelle
    della cartella root, estraendo tau e abeta dal nome del file.
    """
    root_folder = Path(root_folder_path)
    data = []

    # regex flessibile per tau e abeta
    pattern = re.compile(r"tau[_]?(\d+)[_]?abeta[_]?(\d+)", re.IGNORECASE)

    # glob ricorsivo dalla root (trova tutto)
    for file in root_folder.rglob(files_name):
        match = pattern.search(file.name)
        if match:
            tau_val = int(match.group(1))
            abeta_val = int(match.group(2))
        else:
            tau_val = None
            abeta_val = None

        data.append({
            "tau": tau_val,
            "abeta": abeta_val,
            "folder_name": file.parent.name,  # cartella immediata
            "file_name": file.name,
            "file_path": str(file.resolve())
        })

    df = pd.DataFrame(data)
    return df


root_folder = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Stages_08_01_2026"
alpha_band_files = collect_files(root_folder, files_name="FC_EEG alpha band*")
delta_band_files = collect_files(root_folder, files_name="FC_EEG delta band*")
theta_band_files = collect_files(root_folder, files_name="FC_EEG theta band*")
broad_band_files = collect_files(root_folder, files_name="FC_EEG_*")
PSD_files = collect_files(root_folder, files_name="PSD_EEG_*")

tau_stages = 6
abeta_stages = 6
gfp = np.empty((4, tau_stages, abeta_stages)) # Global Field Power
fc_regular = np.empty((4, tau_stages, abeta_stages), dtype=object) #bands, tau_stages, abeta_stages
fc_reduced = np.empty((4, tau_stages, abeta_stages), dtype=object)
psd_means = np.empty((4, tau_stages, abeta_stages))
psd_max = np.empty((4, tau_stages, abeta_stages))
freq_max = np.empty((4, tau_stages, abeta_stages))
freqs = np.empty((tau_stages, abeta_stages), dtype=object)
mean_reg_EEG_array = np.ones((4, tau_stages, abeta_stages))
mean_abs_reg_EEG_array = np.ones((4, tau_stages, abeta_stages))
mean_red_EEG_array = np.ones((4, tau_stages, abeta_stages))
mean_abs_red_EEG_array = np.ones((4, tau_stages, abeta_stages))
for tau in range(tau_stages):
    for abeta in range(abeta_stages):
        fc_b = broad_band_files[(broad_band_files["tau"] == tau) & (broad_band_files["abeta"] == abeta)]
        fc_path_b = fc_b["file_path"]
        fc_string_b = fc_path_b.iloc[0] 
        fc_mat_b = np.load(fc_string_b)

        fc_a = alpha_band_files[(alpha_band_files["tau"] == tau) & (alpha_band_files["abeta"] == abeta)]
        fc_path_a = fc_a["file_path"]
        fc_string_a = fc_path_a.iloc[0] 
        fc_mat_a = np.load(fc_string_a)

        fc_d = delta_band_files[(delta_band_files["tau"] == tau) & (delta_band_files["abeta"] == abeta)]
        fc_path_d = fc_d["file_path"]
        fc_string_d = fc_path_d.iloc[0] 
        fc_mat_d = np.load(fc_string_d)

        fc_t = theta_band_files[(theta_band_files["tau"] == tau) & (theta_band_files["abeta"] == abeta)]
        fc_path_t = fc_t["file_path"]
        fc_string_t = fc_path_t.iloc[0] 
        fc_mat_t = np.load(fc_string_t)

        fc_regular[0, tau, abeta] = fc_mat_b['normal_size']
        fc_regular[1, tau, abeta] = fc_mat_a['normal_size']
        fc_regular[2, tau, abeta] = fc_mat_d['normal_size']
        fc_regular[3, tau, abeta] = fc_mat_t['normal_size']
        mean_reg_EEG_array[0, tau, abeta] = np.mean(fc_regular[0, tau, abeta])
        mean_reg_EEG_array[1, tau, abeta] = np.mean(fc_regular[1, tau, abeta])
        mean_reg_EEG_array[2, tau, abeta] = np.mean(fc_regular[2, tau, abeta])
        mean_reg_EEG_array[3, tau, abeta] = np.mean(fc_regular[3, tau, abeta])
        mean_abs_reg_EEG_array[0, tau, abeta] = np.mean(np.abs(fc_regular[0, tau, abeta]))
        mean_abs_reg_EEG_array[1, tau, abeta] = np.mean(np.abs(fc_regular[1, tau, abeta]))
        mean_abs_reg_EEG_array[2, tau, abeta] = np.mean(np.abs(fc_regular[2, tau, abeta]))
        mean_abs_reg_EEG_array[3, tau, abeta] = np.mean(np.abs(fc_regular[3, tau, abeta]))

        fc_reduced[0, tau, abeta] = fc_mat_b['reduced_size']
        fc_reduced[1, tau, abeta] = fc_mat_a['reduced_size']
        fc_reduced[2, tau, abeta] = fc_mat_d['reduced_size']
        fc_reduced[3, tau, abeta] = fc_mat_t['reduced_size']
        mean_red_EEG_array[0, tau, abeta] = np.mean(fc_reduced[0, tau, abeta])
        mean_red_EEG_array[1, tau, abeta] = np.mean(fc_reduced[1, tau, abeta])
        mean_red_EEG_array[2, tau, abeta] = np.mean(fc_reduced[2, tau, abeta])
        mean_red_EEG_array[3, tau, abeta] = np.mean(fc_reduced[3, tau, abeta])
        mean_abs_red_EEG_array[0, tau, abeta] = np.mean(np.abs(fc_reduced[0, tau, abeta]))
        mean_abs_red_EEG_array[1, tau, abeta] = np.mean(np.abs(fc_reduced[1, tau, abeta]))
        mean_abs_red_EEG_array[2, tau, abeta] = np.mean(np.abs(fc_reduced[2, tau, abeta]))
        mean_abs_red_EEG_array[3, tau, abeta] = np.mean(np.abs(fc_reduced[3, tau, abeta]))

        psd = PSD_files[(PSD_files["tau"] == tau) & (PSD_files["abeta"] == abeta)]
        psd_path = psd["file_path"]
        psd_string = psd_path.iloc[0] 
        psd_mat = np.load(psd_string)
        psd_mean = psd_mat['psd_mean']
        freqs = psd_mat['freqs']

        #broad band
        psd_means[0, tau, abeta] = np.mean(psd_mean)
        psd_max[0, tau, abeta] = np.max(psd_mean)
        idx_max = np.argmax(psd_mean)
        freq_max[0, tau, abeta] = freqs[idx_max]

        #delta band
        mask_delta = (freqs < 4)
        psd_means[2, tau, abeta] = np.mean(psd_mean[mask_delta])
        psd_max[2, tau, abeta] = np.max(psd_mean[mask_delta])
        idx_max = np.where(mask_delta)[0][np.argmax(psd_mean[mask_delta])]
        freq_max[2, tau, abeta] = freqs[idx_max]

        #theta band
        mask_theta = (freqs >= 4) & (freqs < 8)
        psd_means[3, tau, abeta] = np.mean(psd_mean[mask_theta])
        psd_max[3, tau, abeta] = np.max(psd_mean[mask_theta])
        idx_max = np.where(mask_theta)[0][np.argmax(psd_mean[mask_theta])]
        freq_max[3, tau, abeta] = freqs[idx_max]

        #alpha band
        mask_alpha = (freqs >= 8) & (freqs < 12)
        psd_means[1, tau, abeta] = np.mean(psd_mean[mask_alpha])
        psd_max[1, tau, abeta] = np.max(psd_mean[mask_alpha])
        idx_max = np.where(mask_alpha)[0][np.argmax(psd_mean[mask_alpha])]
        freq_max[1, tau, abeta] = freqs[idx_max]



stages = np.linspace(0,5,6)
"""
plt.figure(figsize=(10, 12))
plt.plot(stages, mean_reg_EEG_array[0, :, 0], label='reg_b')
plt.plot(stages, mean_reg_EEG_array[1, :, 0], label='reg_a')
plt.plot(stages, mean_reg_EEG_array[2, :, 0], label='reg_d')
plt.plot(stages, mean_reg_EEG_array[3, :, 0], label='reg_t')
plt.legend()
plt.grid()
plt.title('EEG FC reg mean', fontsize=18)
plt.xlabel('Tau stages', fontsize=16)
plt.show()

plt.figure(figsize=(10, 12))
plt.plot(stages, mean_red_EEG_array[0, :, 0], label='red_b')
plt.plot(stages, mean_red_EEG_array[1, :, 0], label='red_a')
plt.plot(stages, mean_red_EEG_array[2, :, 0], label='red_d')
plt.plot(stages, mean_red_EEG_array[3, :, 0], label='red_t')
plt.legend()
plt.grid()
plt.title('EEG FC red mean', fontsize=18)
plt.xlabel('Tau stages', fontsize=16)
plt.show()

plt.figure(figsize=(10, 12))
plt.plot(stages, mean_abs_reg_EEG_array[0, :, 0], label='reg_abs_b')
plt.plot(stages, mean_abs_reg_EEG_array[1, :, 0], label='reg_abs_a')
plt.plot(stages, mean_abs_reg_EEG_array[2, :, 0], label='reg_abs_d')
plt.plot(stages, mean_abs_reg_EEG_array[3, :, 0], label='reg_abs_t')
plt.legend()
plt.grid()
plt.title('EEG FC abs reg mean', fontsize=18)
plt.xlabel('Tau stages', fontsize=16)
plt.show()

plt.figure(figsize=(10, 12))
plt.plot(stages, mean_abs_red_EEG_array[0, :, 0], label='red_abs_b')
plt.plot(stages, mean_abs_red_EEG_array[1, :, 0], label='red_abs_a')
plt.plot(stages, mean_abs_red_EEG_array[2, :, 0], label='red_abs_d')
plt.plot(stages, mean_abs_red_EEG_array[3, :, 0], label='red_abs_t')
plt.legend()
plt.grid()
plt.title('EEG FC abs red mean', fontsize=18)
plt.xlabel('Tau stages', fontsize=16)
plt.show()

plt.figure(figsize=(10, 12))
plt.plot(stages, freq_max[0, :, 0], label='peak_freq_b')
plt.plot(stages, freq_max[1, :, 0], label='peak_freq_a')
plt.plot(stages, freq_max[2, :, 0], label='peak_freq_d')
plt.plot(stages, freq_max[3, :, 0], label='peak_freq_t')
plt.yticks(np.linspace(2,10,17))
plt.xticks(np.linspace(0,5,6))
plt.legend()
plt.grid('minor')
plt.title('Frequency peaks', fontsize=18)
plt.xlabel('Tau stages', fontsize=16)
plt.show()

plt.figure(figsize=(10, 12))
plt.plot(stages, psd_max[0, :, 0], label='peak_amp_b')
plt.plot(stages, psd_max[1, :, 0], label='peak_amp_a')
plt.plot(stages, psd_max[2, :, 0], label='peak_amp_d')
plt.plot(stages, psd_max[3, :, 0], label='peak_amp_t')
plt.yticks(np.linspace(0,0.008,17))
plt.xticks(np.linspace(0,5,6))
plt.legend()
plt.grid('minor')
plt.title('Peaks amplitude', fontsize=18)
plt.xlabel('Tau stages', fontsize=16)
plt.show()

plt.figure(figsize=(10, 12))
plt.plot(stages, psd_means[0, :, 0], label='peak_amp_b')
plt.plot(stages, psd_means[1, :, 0], label='peak_amp_a')
plt.plot(stages, psd_means[2, :, 0], label='peak_amp_d')
plt.plot(stages, psd_means[3, :, 0], label='peak_amp_t')
plt.yticks(np.linspace(0,0.008,17))
plt.xticks(np.linspace(0,5,6))
plt.legend()
plt.grid('minor')
plt.title('PSD means', fontsize=18)
plt.xlabel('Tau stages', fontsize=16)
plt.show()

plt.figure(figsize=(10, 12))
plt.plot(stages, mean_reg_EEG_array[0, 0, :], label='reg_b')
plt.plot(stages, mean_reg_EEG_array[1, 0, :], label='reg_a')
plt.plot(stages, mean_reg_EEG_array[2, 0, :], label='reg_d')
plt.plot(stages, mean_reg_EEG_array[3, 0, :], label='reg_t')
plt.legend()
plt.grid()
plt.title('EEG FC reg mean', fontsize=18)
plt.xlabel('Abeta stages', fontsize=16)
plt.show()

plt.figure(figsize=(10, 12))
plt.plot(stages, mean_red_EEG_array[0, 0, :], label='red_b')
plt.plot(stages, mean_red_EEG_array[1, 0, :], label='red_a')
plt.plot(stages, mean_red_EEG_array[2, 0, :], label='red_d')
plt.plot(stages, mean_red_EEG_array[3, 0, :], label='red_t')
plt.legend()
plt.grid()
plt.title('EEG FC red mean', fontsize=18)
plt.xlabel('Abeta stages', fontsize=16)
plt.show()

plt.figure(figsize=(10, 12))
plt.plot(stages, mean_abs_reg_EEG_array[0, 0, :], label='reg_abs_b')
plt.plot(stages, mean_abs_reg_EEG_array[1, 0, :], label='reg_abs_a')
plt.plot(stages, mean_abs_reg_EEG_array[2, 0, :], label='reg_abs_d')
plt.plot(stages, mean_abs_reg_EEG_array[3, 0, :], label='reg_abs_t')
plt.legend()
plt.grid()
plt.title('EEG FC abs reg mean', fontsize=18)
plt.xlabel('Abeta stages', fontsize=16)
plt.show()

plt.figure(figsize=(10, 12))
plt.plot(stages, mean_abs_red_EEG_array[0, 0, :], label='red_abs_b')
plt.plot(stages, mean_abs_red_EEG_array[1, 0, :], label='red_abs_a')
plt.plot(stages, mean_abs_red_EEG_array[2, 0, :], label='red_abs_d')
plt.plot(stages, mean_abs_red_EEG_array[3, 0, :], label='red_abs_t')
plt.legend()
plt.grid()
plt.title('EEG FC abs red mean', fontsize=18)
plt.xlabel('Abeta stages', fontsize=16)
plt.show()

plt.figure(figsize=(10, 12))
plt.plot(stages, freq_max[0, 0, :], label='peak_freq_b')
plt.plot(stages, freq_max[1, 0, :], label='peak_freq_a')
plt.plot(stages, freq_max[2, 0, :], label='peak_freq_d')
plt.plot(stages, freq_max[3, 0, :], label='peak_freq_t')
plt.yticks(np.linspace(2,15,27))
plt.xticks(np.linspace(0,5,6))
plt.legend()
plt.grid('minor')
plt.title('Frequency peaks', fontsize=18)
plt.xlabel('Abeta stages', fontsize=16)
plt.show()

plt.figure(figsize=(10, 12))
plt.plot(stages, psd_max[0, 0, :], label='peak_amp_b')
plt.plot(stages, psd_max[1, 0, :], label='peak_amp_a')
plt.plot(stages, psd_max[2, 0, :], label='peak_amp_d')
plt.plot(stages, psd_max[3, 0, :], label='peak_amp_t')
plt.yticks(np.linspace(0,0.008,17))
plt.xticks(np.linspace(0,5,6))
plt.legend()
plt.grid('minor')
plt.title('Peaks amplitude', fontsize=18)
plt.xlabel('Abeta stages', fontsize=16)
plt.show()

plt.figure(figsize=(10, 12))
plt.plot(stages, psd_means[0, 0, :], label='psd_mean_b')
plt.plot(stages, psd_means[1, 0, :], label='psd_mean_a')
plt.plot(stages, psd_means[2, 0, :], label='psd_mean_d')
plt.plot(stages, psd_means[3, 0, :], label='psd_mean_t')
plt.yticks(np.linspace(0,0.0045,10))
plt.xticks(np.linspace(0,5,6))
plt.legend()
plt.grid('minor')
plt.title('PSD means', fontsize=18)
plt.xlabel('Abeta stages', fontsize=16)
plt.show()

fig = plt.figure(figsize=(12, 8)) 
ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
im = ax.imshow(psd_means[0,:,:], cmap='bwr', vmin=None, vmax=None, aspect='equal')
title_text = 'broad PSD mean'
ax.set_title(title_text, fontsize=15)
ax.set_ylabel('Tau Stage', fontsize=12)    # Secondo indice (righe)
ax.set_xlabel('Abeta Stage', fontsize=12)  # Terzo indice (colonne)
cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
plt.colorbar(im, cax=cax)
plt.show()

fig = plt.figure(figsize=(12, 8)) 
ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
im = ax.imshow(psd_means[1,:,:], cmap='bwr', vmin=None, vmax=None, aspect='equal')
title_text = 'alpha PSD mean'
ax.set_title(title_text, fontsize=15)
ax.set_ylabel('Tau Stage', fontsize=12)
ax.set_xlabel('Abeta Stage', fontsize=12)
cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
plt.colorbar(im, cax=cax)
plt.show()

fig = plt.figure(figsize=(12, 8)) 
ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
im = ax.imshow(psd_means[2,:,:], cmap='bwr', vmin=None, vmax=None, aspect='equal')
title_text = 'delta PSD mean'
ax.set_title(title_text, fontsize=15)
ax.set_ylabel('Tau Stage', fontsize=12)
ax.set_xlabel('Abeta Stage', fontsize=12)
cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
plt.colorbar(im, cax=cax)

plt.show()
fig = plt.figure(figsize=(12, 8)) 
ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
im = ax.imshow(psd_means[3,:,:], cmap='bwr', vmin=None, vmax=None, aspect='equal')
title_text = 'theta PSD mean'
ax.set_title(title_text, fontsize=15)
ax.set_ylabel('Tau Stage', fontsize=12)
ax.set_xlabel('Abeta Stage', fontsize=12)
cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
plt.colorbar(im, cax=cax)
plt.show()
"""

fig = plt.figure(figsize=(12, 8)) 
ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
im = ax.imshow(mean_reg_EEG_array[0,:,:], cmap='bwr', vmin=None, vmax=None, aspect='equal') 
title_text = 'broad FC mean reg'
ax.set_title(title_text, fontsize=15)
ax.set_ylabel('Tau Stage', fontsize=12)
ax.set_xlabel('Abeta Stage', fontsize=12)
cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
plt.colorbar(im, cax=cax)
plt.show()

fig = plt.figure(figsize=(12, 8)) 
ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
im = ax.imshow(mean_reg_EEG_array[1,:,:], cmap='bwr', vmin=None, vmax=None, aspect='equal') 
title_text = 'alpha FC mean reg'
ax.set_title(title_text, fontsize=15)
ax.set_ylabel('Tau Stage', fontsize=12)
ax.set_xlabel('Abeta Stage', fontsize=12)
cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
plt.colorbar(im, cax=cax)
plt.show()

fig = plt.figure(figsize=(12, 8)) 
ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
im = ax.imshow(mean_reg_EEG_array[2,:,:], cmap='bwr', vmin=None, vmax=None, aspect='equal') 
title_text = 'delta FC mean reg'
ax.set_title(title_text, fontsize=15)
ax.set_ylabel('Tau Stage', fontsize=12)
ax.set_xlabel('Abeta Stage', fontsize=12)
cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
plt.colorbar(im, cax=cax)
plt.show()

fig = plt.figure(figsize=(12, 8)) 
ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
im = ax.imshow(mean_reg_EEG_array[3,:,:], cmap='bwr', vmin=None, vmax=None, aspect='equal') 
title_text = 'theta FC mean reg'
ax.set_title(title_text, fontsize=15)
ax.set_ylabel('Tau Stage', fontsize=12)
ax.set_xlabel('Abeta Stage', fontsize=12)
cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
plt.colorbar(im, cax=cax)
plt.show()

fig = plt.figure(figsize=(12, 8)) 
ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
im = ax.imshow(mean_red_EEG_array[0,:,:], cmap='bwr', vmin=None, vmax=None, aspect='equal') 
title_text = 'broad FC mean red'
ax.set_title(title_text, fontsize=15)
ax.set_ylabel('Tau Stage', fontsize=12)
ax.set_xlabel('Abeta Stage', fontsize=12)
cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
plt.colorbar(im, cax=cax)
plt.show()

fig = plt.figure(figsize=(12, 8)) 
ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
im = ax.imshow(mean_red_EEG_array[1,:,:], cmap='bwr', vmin=None, vmax=None, aspect='equal')
title_text = 'alpha FC mean red'
ax.set_title(title_text, fontsize=15)
ax.set_ylabel('Tau Stage', fontsize=12)
ax.set_xlabel('Abeta Stage', fontsize=12)
cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
plt.colorbar(im, cax=cax)
plt.show()

fig = plt.figure(figsize=(12, 8)) 
ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
im = ax.imshow(mean_red_EEG_array[2,:,:], cmap='bwr', vmin=None, vmax=None, aspect='equal') 
title_text = 'delta FC mean red'
ax.set_title(title_text, fontsize=15)
ax.set_ylabel('Tau Stage', fontsize=12)
ax.set_xlabel('Abeta Stage', fontsize=12)
cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
plt.colorbar(im, cax=cax)
plt.show()

fig = plt.figure(figsize=(12, 8)) 
ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
im = ax.imshow(mean_red_EEG_array[3,:,:], cmap='bwr', vmin=None, vmax=None, aspect='equal') 
title_text = 'theta FC mean red'
ax.set_title(title_text, fontsize=15)
ax.set_ylabel('Tau Stage', fontsize=12)
ax.set_xlabel('Abeta Stage', fontsize=12)
cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
plt.colorbar(im, cax=cax)
plt.show()

fig = plt.figure(figsize=(12, 8)) 
ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
im = ax.imshow(mean_abs_reg_EEG_array[0,:,:], cmap='bwr', vmin=None, vmax=None, aspect='equal')
title_text = 'broad abs FC mean reg'
ax.set_title(title_text, fontsize=15)
ax.set_ylabel('Tau Stage', fontsize=12)
ax.set_xlabel('Abeta Stage', fontsize=12)
cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
plt.colorbar(im, cax=cax)
plt.show()

fig = plt.figure(figsize=(12, 8)) 
ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
im = ax.imshow(mean_abs_reg_EEG_array[1,:,:], cmap='bwr', vmin=None, vmax=None, aspect='equal')  
title_text = 'alpha abs FC mean reg'
ax.set_title(title_text, fontsize=15)
ax.set_ylabel('Tau Stage', fontsize=12)
ax.set_xlabel('Abeta Stage', fontsize=12)
cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
plt.colorbar(im, cax=cax)
plt.show()

fig = plt.figure(figsize=(12, 8)) 
ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
im = ax.imshow(mean_abs_reg_EEG_array[2,:,:], cmap='bwr', vmin=None, vmax=None, aspect='equal')  
title_text = 'delta abs FC mean reg'
ax.set_title(title_text, fontsize=15)
ax.set_ylabel('Tau Stage', fontsize=12)
ax.set_xlabel('Abeta Stage', fontsize=12)
cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
plt.colorbar(im, cax=cax)
plt.show()

fig = plt.figure(figsize=(12, 8)) 
ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
im = ax.imshow(mean_abs_reg_EEG_array[3,:,:], cmap='bwr', vmin=None, vmax=None, aspect='equal')  
title_text = 'theta abs FC mean reg'
ax.set_title(title_text, fontsize=15)
ax.set_ylabel('Tau Stage', fontsize=12)
ax.set_xlabel('Abeta Stage', fontsize=12)
cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
plt.colorbar(im, cax=cax)
plt.show()

fig = plt.figure(figsize=(12, 8)) 
ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
im = ax.imshow(mean_abs_red_EEG_array[0,:,:], cmap='bwr', vmin=None, vmax=None, aspect='equal') 
title_text = 'broad abs FC mean red'
ax.set_title(title_text, fontsize=15)
ax.set_ylabel('Tau Stage', fontsize=12)
ax.set_xlabel('Abeta Stage', fontsize=12)
cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
plt.colorbar(im, cax=cax)
plt.show()

fig = plt.figure(figsize=(12, 8)) 
ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
im = ax.imshow(mean_abs_red_EEG_array[1,:,:], cmap='bwr', vmin=None, vmax=None, aspect='equal') 
title_text = 'alpha abs FC mean red'
ax.set_title(title_text, fontsize=15)
ax.set_ylabel('Tau Stage', fontsize=12)
ax.set_xlabel('Abeta Stage', fontsize=12)
cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
plt.colorbar(im, cax=cax)
plt.show()

fig = plt.figure(figsize=(12, 8)) 
ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
im = ax.imshow(mean_abs_red_EEG_array[2,:,:], cmap='bwr', vmin=None, vmax=None, aspect='equal') 
title_text = 'delta asb FC mean red'
ax.set_title(title_text, fontsize=15)
ax.set_ylabel('Tau Stage', fontsize=12)
ax.set_xlabel('Abeta Stage', fontsize=12)
cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
plt.colorbar(im, cax=cax)
plt.show()

fig = plt.figure(figsize=(12, 8)) 
ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
im = ax.imshow(mean_abs_red_EEG_array[3,:,:], cmap='bwr', vmin=None, vmax=None, aspect='equal')  
title_text = 'theta abs FC mean red'
ax.set_title(title_text, fontsize=15)
ax.set_ylabel('Tau Stage', fontsize=12)
ax.set_xlabel('Abeta Stage', fontsize=12)
cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
plt.colorbar(im, cax=cax)
plt.show()
import pandas as pd
from pathlib import Path
import re
import os
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

def file_extraction(files_group, tau_stage, abeta_stage):
    file_sel = files_group[(files_group["tau"] == tau_stage) & (files_group["abeta"] == abeta_stage)]
    file_path_sel = file_sel["file_path"]
    file_path_string = file_path_sel.iloc[0] 
    file_data = np.load(file_path_string)
    return file_data

def stage_metrics(root_folder = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Stages_21_01_2026_img_coh"):
    alpha_band_files = collect_files(root_folder, files_name="FC_EEG alpha band*")
    beta_band_files = collect_files(root_folder, files_name="FC_EEG beta band*")
    delta_band_files = collect_files(root_folder, files_name="FC_EEG delta band*")
    theta_band_files = collect_files(root_folder, files_name="FC_EEG theta band*")
    broad_band_files = collect_files(root_folder, files_name="FC_EEG_*")
    bands_files = [broad_band_files, delta_band_files, theta_band_files, alpha_band_files, beta_band_files]
    PSD_files = collect_files(root_folder, files_name="PSD_EEG_*")

    tau_stages = 6
    abeta_stages = 6
    psd_means = np.zeros((tau_stages, abeta_stages, 5)) #broad (0.5-45)Hz, delta (0.5-4)Hz, theta (4-8)Hz, alpha (8-12)Hz, beta (12-30)Hz 
    fc_reg_means = np.zeros((tau_stages, abeta_stages, 5, 2)) #tau stage, abeta stage, band, mean or mean absolute value
    fc_red_means = np.zeros((tau_stages, abeta_stages, 5, 2)) #tau stage, abeta stage, band, mean or mean absolute value
    for tau in range(tau_stages):
        for abeta in range(abeta_stages):
            for band in range(len(bands_files)):
                fc_matrix = file_extraction(bands_files[band], tau, abeta)
                fc_reg_means[tau,abeta, band, 0] = np.mean(fc_matrix['normal_size'])
                fc_reg_means[tau,abeta, band, 1] = np.mean(np.abs(fc_matrix['normal_size']))
                fc_red_means[tau,abeta, band, 0] = np.mean(fc_matrix['reduced_size'])
                fc_red_means[tau,abeta, band, 1] = np.mean(np.abs(fc_matrix['reduced_size']))

            psd_matrix = file_extraction(PSD_files, tau, abeta)
            psd = psd_matrix['psd_mean']
            freqs = psd_matrix['freqs']
            psd_means[tau, abeta, 0] = np.mean(psd)                               #broad
            psd_means[tau, abeta, 1] = np.mean(psd[(freqs < 4)])                  #delta
            psd_means[tau, abeta, 2] = np.mean(psd[(freqs >= 4) & (freqs < 8)])   #theta
            psd_means[tau, abeta, 3] = np.mean(psd[(freqs >= 8) & (freqs < 12)])  #alpha
            psd_means[tau, abeta, 4] = np.mean(psd[(freqs >= 12) & (freqs < 30)]) #beta
    return fc_reg_means, fc_red_means, psd_means

fc_reg_means, fc_red_means, psd_means = stage_metrics()
data_save_dir = os.path.join(gv.data_folder, f"Stages_21_01_2026_img_coh")
os.makedirs(data_save_dir, exist_ok=True)
output = os.path.join(data_save_dir, f"stage_metrics.npz")
np.savez_compressed(
            output,
            fc_reg_means=fc_reg_means,
            fc_red_means=fc_red_means,
            psd_means=psd_means
        )
root_folder = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Stages_21_01_2026_img_coh"
alpha_band_files = collect_files(root_folder, files_name="FC_EEG alpha band*")
delta_band_files = collect_files(root_folder, files_name="FC_EEG delta band*")
theta_band_files = collect_files(root_folder, files_name="FC_EEG theta band*")
beta_band_files = collect_files(root_folder, files_name="FC_EEG beta band*")
broad_band_files = collect_files(root_folder, files_name="FC_EEG_*")
PSD_files = collect_files(root_folder, files_name="PSD_EEG_*")

def file_extraction(files_group, tau_stage, abeta_stage):
    file_sel = files_group[(files_group["tau"] == tau_stage) & (files_group["abeta"] == abeta_stage)]
    file_path_sel = file_sel["file_path"]
    file_path_string = file_path_sel.iloc[0] 
    file_data = np.load(file_path_string)
    return file_data

bands_files = [broad_band_files, delta_band_files, theta_band_files, alpha_band_files, beta_band_files]

root_folder = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Stages_21_01_2026_img_coh"
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
mean_reg_EEG_array = np.ones((4, tau_stages, abeta_stages)) #bands, tau, abeta
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

def plot_function(x, title, xlabel, labels=['broad','delta','theta', 'alpha', 'beta'], y = np.linspace(0,5,6), tau = True):
    plt.figure(figsize=(10, 12))
    for i in range(len(labels)):
        if tau == True:
            plt.plot(y, x[i, :, 0], label=labels[i])
        else:
            plt.plot(y, x[i, 0, :], label=labels[i])
    plt.legend()
    plt.grid()
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.show()

def plot_heatmap(x, title, idx):
    fig = plt.figure(figsize=(12, 8)) 
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
    im = ax.imshow(x[:,:,idx], cmap='bwr', vmin=None, vmax=None, aspect='equal')
    title_text = title
    ax.set_title(title_text, fontsize=15)
    ax.set_ylabel('Tau Stage', fontsize=12)# Primo indice (righe)
    ax.set_xlabel('Abeta Stage', fontsize=12)  # Secondo indice (colonne)
    cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
    plt.colorbar(im, cax=cax)
    plt.show()

def plot_heatmap2(x, title, idx, abs = 0):
    fig = plt.figure(figsize=(12, 8)) 
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
    im = ax.imshow(x[idx,:,:], cmap='bwr', vmin=None, vmax=None, aspect='equal')
    title_text = title
    ax.set_title(title_text, fontsize=15)
    ax.set_ylabel('Tau Stage', fontsize=12)    # Primo indice (righe)
    ax.set_xlabel('Abeta Stage', fontsize=12)  # Secondo indice (colonne)
    cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
    plt.colorbar(im, cax=cax)
    plt.show()

def plot_heatmaps():
    """
    plot_heatmap(x=psd_means, title='broad PSD mean', idx = 0)
    plot_heatmap(x=psd_means, title='delta PSD mean', idx = 1)
    plot_heatmap(x=psd_means, title='theta PSD mean', idx = 2)
    plot_heatmap(x=psd_means, title='alpha PSD mean', idx = 3)
    plot_heatmap(x=psd_means, title='beta PSD mean', idx = 4)
    """
    plot_heatmap(x=fc_reg_means[:, :, :, 0], title='broad FC mean reg', idx = 0) #tau stage, abeta stage, band, mean or mean absolute value
    plot_heatmap2(x=mean_reg_EEG_array[:, :, :], title='broad FC mean reg', idx = 0)  #bands, tau, abeta
    #plot_heatmap(x=fc_reg_means[:, :, :, 0], title='delta FC mean reg', idx = 1)
    #plot_heatmap(x=fc_reg_means[:, :, :, 0], title='theta FC mean reg', idx = 2)
    #plot_heatmap(x=fc_reg_means[:, :, :, 0], title='alpha FC mean reg', idx = 3)
    #plot_heatmap(x=fc_reg_means[:, :, :, 0], title='beta FC mean reg', idx = 4)
    
    #plot_heatmap2(x=mean_reg_EEG_array[:, :, :], title='delta FC mean reg', idx = 1)
    #plot_heatmap2(x=mean_reg_EEG_array[:, :, :], title='theta FC mean reg', idx = 2)
    #plot_heatmap2(x=mean_reg_EEG_array[:, :, :], title='alpha FC mean reg', idx = 3)
    #plot_heatmap2(x=mean_reg_EEG_array[:, :, :], title='beta FC mean reg', idx = 4)
    """
    plot_heatmap(x=fc_red_means[:, :, :, 0], title='broad FC mean red', idx = 0)
    plot_heatmap(x=fc_red_means[:, :, :, 0], title='delta FC mean red', idx = 1)
    plot_heatmap(x=fc_red_means[:, :, :, 0], title='theta FC mean red', idx = 2)
    plot_heatmap(x=fc_red_means[:, :, :, 0], title='alpha FC mean red', idx = 3)
    plot_heatmap(x=fc_red_means[:, :, :, 0], title='beta FC mean red', idx = 4)
    plot_heatmap(x=fc_reg_means[:, :, :, 1], title='broad abs FC mean reg', idx = 0)
    plot_heatmap(x=fc_reg_means[:, :, :, 1], title='delta abs FC mean reg', idx = 1)
    plot_heatmap(x=fc_reg_means[:, :, :, 1], title='theta abs FC mean reg', idx = 2)
    plot_heatmap(x=fc_reg_means[:, :, :, 1], title='alpha abs FC mean reg', idx = 3)
    plot_heatmap(x=fc_reg_means[:, :, :, 1], title='beta abs FC mean reg', idx = 4)
    plot_heatmap(x=fc_red_means[:, :, :, 1], title='broad abs FC mean red', idx = 0)
    plot_heatmap(x=fc_red_means[:, :, :, 1], title='delta abs FC mean red', idx = 1)
    plot_heatmap(x=fc_red_means[:, :, :, 1], title='theta abs FC mean red', idx = 2)
    plot_heatmap(x=fc_red_means[:, :, :, 1], title='alpha abs FC mean red', idx = 3)
    plot_heatmap(x=fc_red_means[:, :, :, 1], title='beta abs FC mean red', idx = 4)
    """

def plot_functions():
    plot_function(x = fc_reg_means[:, :, :, 0], title = 'EEG FC reg mean', xlabel='Tau stages')
    plot_function(x = fc_red_means[:, :, :, 0], title = 'EEG FC red mean', xlabel='Tau stages')
    plot_function(x = fc_reg_means[:, :, :, 1], title = 'EEG FC abs reg mean', xlabel='Tau stages')
    plot_function(x = fc_red_means[:, :, :, 1], title = 'EEG FC abs red mean', xlabel='Tau stages')
    #plot_function(x = freq_max, title = 'Frequency peaks', xlabel='Tau stages')
    #plot_function(x = psd_max, title = 'Frequency amplitude', xlabel='Tau stages')
    plot_function(x = psd_means, title = 'PSD mean', xlabel='Tau stages')

plot_heatmaps()
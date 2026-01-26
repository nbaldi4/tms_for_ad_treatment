import pandas as pd
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import global_variable_creation as gv

from data_analysis_utils import real_vs_stages_PSD, real_vs_stages_FC, to_common_indices, macro_fc, analyze_regular_data, analyze_reduced_data


# Rimuove il limite di larghezza della colonna (None = nessun limite)
pd.set_option('display.max_colwidth', None)

abeta_MCI_patients = np.array([1559.0, np.nan, 338.0, 974.0, np.nan, 384.0, np.nan, 1239.0, 758.0, np.nan, np.nan, 1742.0,
                               np.nan, 527.0, np.nan, 702.0, 1170.0, np.nan, 812.0, 1238.0, np.nan, np.nan, 1202.0, 702.0,
                               np.nan, 692.0, 876.0, 1251.0, 490.0, np.nan, 1320.0, 1962.0, np.nan, 1374.0, np.nan, 1572.0,
                               685.0, 698.0, np.nan, 464.0])
tau_MCI_patients = np.array([40.0, np.nan, 128.2, 20.0, np.nan, 30.6, np.nan, 44.0, 125.0, np.nan, np.nan, 53.0, np.nan, 34.0,
                             np.nan, 81.0, 28.0, np.nan, 56.0, 30.0, np.nan, np.nan, 25.0, 40.0, np.nan, 103.0, 158.0, 36.0,
                             103.0, np.nan, 39.4, 102.6, np.nan, 40.4, np.nan, 42.9, 156.9, 83.0, np.nan, 18.2])



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

def patient_extraction(files_group, tau_stage, abeta_stage):
    file_sel = files_group[(files_group["tau"] == tau_stage) & (files_group["abeta"] == abeta_stage)]
    file_path_sel = file_sel["file_path"]
    file_path_string = file_path_sel.iloc[0] 
    file_data = np.load(file_path_string)
    return file_data

def stage_metrics(PSD_patient, FC_patient, root_folder):
    root_folder = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Stages_21_01_2026_img_coh"
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
            for band in range(bands_files):
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

patient_files = collect_files(root_folder_path=gv.data_folder, files_name="patient_*")

def stage_identification(fc_reg_pat, psd_pat): #pat for patient (not for path)
    tau_stages = 6
    abeta_stages = 6
    distances = np.zeros((tau_stages, abeta_stages, 15)) #5 for fc_reg, 5 for fc_red, 5 for PSD

fc_healty = np.load(gv.MCI_FC_data)
psd_healty = np.load(gv.MCI_PSD_data)
nearest = np.argmin(distances)
nearest_tau_stage = np.floor(nearest/tau_stages)
nearest_abeta_stage = nearest - nearest_tau_stage*abeta_stages
print("tau = ", nearest_tau_stage, "; abeta = ", nearest_abeta_stage, "; CSF tau = ", tau_MCI_patients[i], "; CSF abeta = ", abeta_MCI_patients[i])

"""
fc_healty = np.load(gv.HC_FC_data)
print(fc_regular[0,0].shape)
fc_reduced_stage=macro_fc(fc_regular[0,0])
print(fc_reduced_stage.shape)
fc_mean = np.mean(fc_healty, axis=0)
print(fc_mean.shape)
matrix1, matrix2 = to_common_indices(fc_regular[0,0], fc_mean)
print(matrix1.shape)
print(matrix2.shape)
print('np.max(fc_mean)',np.max(fc_mean))
for i in range(len(fc_healty[:,0,0])):
    print(i)
    print(np.max(fc_healty[i,:,:]))

fc = fc_regular[0,0]

print(fc_healty.shape)
print(fc_regular[0,0].shape)
sub_fc_healty, com_fc_healty = [], []
for i in range(len(fc_healty[:, 0, 0])):
    matrix1, matrix2 = to_common_indices(fc_regular[0,0], fc_healty[i, :, :])
    com_fc_healty.append(matrix2)

fc_regular_0_0 = matrix1
com_fc_healty = np.array(com_fc_healty)
com_fc_healty = np.clip(com_fc_healty, -0.999999, 0.999999)

# --- Compute group-level statistics ---
normal_corr_stack = np.stack(com_fc_healty, axis=0)
normal_z_stack = np.arctanh(normal_corr_stack)
normal_z_mean = np.mean(normal_z_stack, axis=0)
normal_corr_mean = np.tanh(normal_z_mean)


gv.show_plots = True
norm_mae, red_mae = real_vs_stages_FC(fc_real=fc_mean, fc_regular_stage=fc_regular[0,0], fc_reduced_stage=macro_fc(fc_regular[0,0]), return_value=True)
real_vs_stages_PSD(np.load(gv.HC_PSD_data), stage_psd=psd[0,0], stage_freqs=freqs[0,0])
print('norm_mae',norm_mae)
print('red_mae',red_mae)
print('psd_mae',psd_maes[0,0])
"""
"""
print(f"Shape matrice normale: {fc_regular.shape}")
print(f"Shape matrice normale: {psd.shape}")
print(fc_regular[0,0])
print(psd[5,0].shape[0])
freqs_uniform = np.linspace(0, psd[5,0].shape[0]-1, psd[0,0].shape[0])
plt.figure(figsize=(10, 6))
plt.plot(freqs_uniform, psd[5,0])
plt.ylim(0, 1.01)
plt.xlim(0, 45)
plt.xlabel('Frequency [Hz]', fontsize=16)
plt.ylabel('Normalized PSD', fontsize=16)
plt.grid(True)
plt.show()
"""
#for row in alpha_band_files.itertuples():
#    print(row.tau, row.abeta, row.file_path)
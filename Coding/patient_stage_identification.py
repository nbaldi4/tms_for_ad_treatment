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
abeta_stages = 1
fc_regular = np.empty((tau_stages, abeta_stages), dtype=object)
psd = np.empty((tau_stages, abeta_stages), dtype=object)
freqs = np.empty((tau_stages, abeta_stages), dtype=object)
psd_maes = np.zeros((tau_stages, abeta_stages))
fc_reg_maes = np.zeros((tau_stages, abeta_stages))
fc_red_maes = np.zeros((tau_stages, abeta_stages))
fc_healty = np.load(gv.MCI_FC_data)
psd_healty = np.load(gv.MCI_PSD_data)
for i in [26]:#range(len(psd_healty)):
    fc_mean = fc_healty[i]#np.mean(fc_healty, axis=0)
    psd_mean = psd_healty[i]
    for tau in range(tau_stages):
        for abeta in range(abeta_stages):
            fc_sel = broad_band_files[(broad_band_files["tau"] == tau) & (broad_band_files["abeta"] == abeta)]
            fc_path_sel = fc_sel["file_path"]
            fc_path_string = fc_path_sel.iloc[0] 
            fc_matrix = np.load(fc_path_string)
            fc_regular[tau, abeta] = fc_matrix['normal_size']
            psd_sel = PSD_files[(PSD_files["tau"] == tau) & (PSD_files["abeta"] == abeta)]
            psd_path_sel = psd_sel["file_path"]
            psd_path_string = psd_path_sel.iloc[0] 
            psd_matrix = np.load(psd_path_string)
            psd[tau, abeta] = psd_matrix['psd_mean']
            freqs[tau, abeta] = psd_matrix['freqs']
            psd_maes[tau,abeta] = real_vs_stages_PSD(psd_mean, stage_psd=psd[tau,abeta], stage_freqs=freqs[tau,abeta], return_value=True)
            fc_reg_maes[tau,abeta], fc_red_maes[tau,abeta] = real_vs_stages_FC(fc_real=fc_mean, fc_regular_stage=fc_regular[tau,abeta], fc_reduced_stage=macro_fc(fc_regular[tau,abeta]), return_value=True)

    psd_mean = np.mean(psd_maes)
    fc_red_mean = np.mean(fc_red_maes)
    fc_reg_mean = np.mean(fc_reg_maes)
    psd_dev = (psd_maes-psd_mean)/psd_mean
    fc_red_dev = (fc_red_maes-fc_red_mean)/fc_red_mean
    fc_reg_dev = (fc_reg_maes-fc_reg_mean)/fc_reg_mean

    distances = fc_red_dev + fc_reg_dev + 2*psd_dev

    #print(distances) 26
    print(np.argmin(distances))

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
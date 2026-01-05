import pandas as pd
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

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


root_folder = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Stages"
alpha_band_files = collect_files(root_folder, files_name="FC_EEG alpha band*")
delta_band_files = collect_files(root_folder, files_name="FC_EEG delta band*")
theta_band_files = collect_files(root_folder, files_name="FC_EEG theta band*")
broad_band_files = collect_files(root_folder, files_name="FC_EEG_*")
PSD_files = collect_files(root_folder, files_name="PSD_EEG_*")

fc_regular = np.empty((6, 1), dtype=object)
psd = np.empty((6, 1), dtype=object)
for tau in range(6):
    for abeta in range(1):
        fc_sel = broad_band_files[(broad_band_files["tau"] == tau) & (alpha_band_files["abeta"] == abeta)]
        fc_path_sel = fc_sel["file_path"]
        fc_path_string = fc_path_sel.iloc[0] 
        fc_matrix = np.load(fc_path_string)
        fc_regular[tau, abeta] = fc_matrix['normal_size']
        psd_sel = PSD_files[(PSD_files["tau"] == tau) & (alpha_band_files["abeta"] == abeta)]
        psd_path_sel = psd_sel["file_path"]
        psd_path_string = psd_path_sel.iloc[0] 
        psd_matrix = np.load(psd_path_string)
        psd[tau, abeta] = psd_matrix['psd_mean']

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

#for row in alpha_band_files.itertuples():
#    print(row.tau, row.abeta, row.file_path)
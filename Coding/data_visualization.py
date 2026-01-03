import numpy as np
import matplotlib.pyplot as plt

# Carica l'archivio
file_path = r'C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\tau5_abeta0_a0.0701_b0.0653_g0.1099_v299.70_n0.1028_simtime20000\FC_EEG alpha band (8-14)Hz_tau_5_abeta_0_band_8-14_Hz.npz'
data = np.load(file_path)
print(data.files)

with np.load(file_path) as data:
    sub_corr = data['reduced_size']
    print(sub_corr)
    sub_corr = 2 * (sub_corr - np.min(sub_corr)) / (np.max(sub_corr) - np.min(sub_corr)) - 1
    corr_sub_mean = np.mean(sub_corr)
    abs_sub_corr_mean = np.mean(np.abs(sub_corr))
    # Plot corr
    fig = plt.figure(figsize=(12, 8)) 
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
    im = ax.imshow(sub_corr, cmap='bwr', vmin=-1, vmax=1, aspect='equal') 
    title_text = f"Reduced size simulated FC: reduced mean = {corr_sub_mean:.4f}, ({abs_sub_corr_mean:.4f})"
    ax.set_title(title_text, fontsize=15)
    cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
    plt.colorbar(im, cax=cax)
    plt.show()

    corr = data['normal_size']
    print(corr)
    corr = 2 * (corr - np.min(corr)) / (np.max(corr) - np.min(corr)) - 1
    corr_mean = np.mean(corr)
    abs_corr_mean = np.mean(np.abs(corr))
    # Plot corr
    fig = plt.figure(figsize=(12, 8)) 
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
    im = ax.imshow(corr, cmap='bwr', vmin=-1, vmax=1, aspect='equal') 
    title_text = f"Regular size simulated FC: reduced mean = {corr_mean:.4f}, ({abs_corr_mean:.4f})"
    ax.set_title(title_text, fontsize=15)
    cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
    plt.colorbar(im, cax=cax)
    plt.show()

import pandas as pd
from pathlib import Path
import re

def collect_FC_EEG_files(root_folder_path):
    """
    Scansiona tutte le sottocartelle di `root_folder_path` e raccoglie tutti i file
    che iniziano con "FC_EEG alpha band", estraendo tau e abeta dal nome del file.
    """
    
    root_folder = Path(root_folder_path)
    data = []

    # pattern per estrarre tau e abeta dal file
    pattern = re.compile(r"_tau_(\d+)_abeta_(\d+)_")

    # ciclo su tutte le sottocartelle
    for subfolder in root_folder.iterdir():
        if subfolder.is_dir():
            # cerca tutti i file che iniziano con "FC_EEG alpha band"
            for file in subfolder.rglob("FC_EEG alpha band*"):
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
                    "folder_name": subfolder.name,
                    "file_name": file.name,
                    "file_path": str(file.resolve())
                })
    
    df = pd.DataFrame(data)
    return df

root_folder = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data"
df_files = collect_FC_EEG_files(root_folder)

print(df_files.head())
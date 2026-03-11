import pandas as pd
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import global_variable_creation as gv
import os

from data_analysis_utils import real_vs_stages_PSD, real_vs_stages_FC, to_common_indices, macro_fc, analyze_regular_data, analyze_reduced_data

root_path = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data"
file_name = "Stages_16_02_N_stages_"

mci_data = np.load(os.path.join(root_path, r"Patients_highlights\mci_summary_stats_ref_freqs.npz"), allow_pickle=True)
mci_fc_abs_mean = mci_data['mci_fc_abs_mean']
mci_psd_rel_pow = mci_data['mci_psd_rel_pow']
mci_csf_values = mci_data['mci_csf']

band_names = ['broad', 'delta', 'theta', 'alpha', 'beta']

for band in range(len(band_names)):
        
    data = mci_fc_abs_mean[:, band]
        
    plt.figure(figsize=(5,4))
    plt.boxplot(data.T)
        
    plt.title(band_names[band] + " band")
    #plt.xlabel(xlabel)
    #plt.ylabel(ylabel)
        
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


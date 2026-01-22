import pandas as pd
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import global_variable_creation as gv
import time

from data_analysis_utils import real_vs_stages_PSD, real_vs_stages_FC, to_common_indices, macro_fc, analyze_regular_data, analyze_reduced_data
import mne
from utils import psd, functional_connectivity_img_coh

from pymatreader import read_mat
import mne

start = time.time()

# Rimuove il limite di larghezza della colonna (None = nessun limite)


# Esempio caricamento soggetto 211 Occhi Chiusi
path_211 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_211_final.set"
path_0 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_1_final.set"
path_1 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_2_final.set"
path_2 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_3_final.set"
path_3 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_4_final.set"
path_4 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_5_final.set"
path_5 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_6_final.set"
path_6 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_7_final.set"
path_7 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_8_final.set"
path_8 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_9_final.set"
path_9 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_10_final.set"
path_10 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_11_final.set"
path_11 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_12_final.set"
path_12 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_13_final.set"
path_13 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_14_final.set"
path_14 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_15_final.set"
path_15 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_16_final.set"
path_16 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_17_final.set"
path_17 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_18_final.set"
path_18 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_19_final.set"
path_19 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_20_final.set"
path_20 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_21_final.set"
path_21 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_22_final.set"
path_22 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_23_final.set"
path_23 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_24_final.set"
path_24 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_25_final.set"
path_25 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_26_final.set"
path_26 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_27_final.set"
path_27 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_28_final.set"
path_28 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_29_final.set"
path_29 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_30_final.set"
path_30 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_31_final.set"
path_31 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_32_final.set"
path_32 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_33_final.set"
path_33 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_34_final.set"
path_34 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_35_final.set"
path_35 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_36_final.set"
path_36 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_37_final.set"
path_37 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_38_final.set"
path_38 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_39_final.set"
path_39 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_40_final.set"
path_40 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_41_final.set"
path_41 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_42_final.set"
path_42 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_43_final.set"
path_43 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_44_final.set"
path_44 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_45_final.set"
path_45 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\EC_EEG_PREVIEW_final\EC_EEG_PREVIEW_46_final.set"

paths = np.array([path_1, path_2, path_3, path_4, path_5, path_6, path_7, path_8, path_9, path_10,
                  path_11, path_12, path_13, path_14, path_15, path_16, path_17, path_18, path_19,
                  path_20, path_21, path_22, path_23, path_24, path_25, path_26, path_27, path_28,
                  path_29, path_30, path_31, path_32, path_33, path_34, path_35, path_36, path_37,
                  path_38, path_39, path_40, path_41, path_42, path_43, path_44, path_45])

abeta_level = np.loadtxt(r"C:\Users\User\OneDrive - University of Pisa\Desktop\CSF_abeta.csv", skiprows=1)
tau_level = np.loadtxt(r"C:\Users\User\OneDrive - University of Pisa\Desktop\CSF_p_tau.csv", skiprows=1)
prot_lev = np.column_stack((abeta_level, tau_level))
print(prot_lev[0])
gv.save_plots = True

def fun(path_ec, proteins, patient_num):

    data_dict = read_mat(path_ec)

    if 'EEG' in data_dict: # finds data marked as 'EEG'
        eeg_struct = data_dict['EEG']
    else:
        eeg_struct = data_dict

    data = eeg_struct['data'] # collect EEG data

    # if data is 3D (epochs), reshape them in 2D
    if data.ndim == 3:
        n_epochs, n_channels, n_samples = data.shape
        data = data.transpose(1, 0, 2).reshape(n_channels, -1)

    # EEGLAB uses microvolt (uV), MNE uses Volt (V)
    data = data * 1e-6 

    # estract sample frequency and channels name
    sfreq = eeg_struct['srate']

    # 'chanlocs' and 'chanlist' are two possibile names under which channels can be found
    if 'chanlocs' in eeg_struct and isinstance(eeg_struct['chanlocs'], dict):
        ch_names = eeg_struct['chanlocs']['labels']
    elif 'chanlocs' in eeg_struct and isinstance(eeg_struct['chanlocs'], list):
        ch_names = [c['labels'] for c in eeg_struct['chanlocs']]
    else:
        # attach fake names if not present
        ch_names = [f'CH{i:03d}' for i in range(data.shape[0])]

    info = mne.create_info(ch_names=list(ch_names), sfreq=sfreq, ch_types='eeg')
    raw_ec = mne.io.RawArray(data, info)

    #print(f"Caricamento completato! Canali rilevati: {len(ch_names)}")
    #print(f"Frequenza di campionamento: {sfreq} Hz")
    eeg_data = raw_ec.get_data() 

    # reshape in TVB notation
    eeg_reshape = eeg_data.T[10000:, np.newaxis, :, np.newaxis]

    # sample frequency
    gv.temp_avg_period = 1000 / raw_ec.info['sfreq']
    return eeg_reshape


def analyze_patients(paths, prot_lev):
    for i in range(len(paths)):
        eeg = fun(paths[i], prot_lev[i], patient_num=i+1)
        functional_connectivity_img_coh(time_series_original = eeg, label = f'EEG broad band (0.5-45)Hz', lowcut=0.5, highcut=45, real_patient=True, patient=i+1)
        functional_connectivity_img_coh(time_series_original = eeg, label = f'EEG beta band (12-30)Hz', lowcut=12, highcut=30, real_patient=True, patient=i+1)
        functional_connectivity_img_coh(time_series_original = eeg, label = f'EEG alpha band (8-12)Hz', lowcut=8, highcut=12, real_patient=True, patient=i+1)
        functional_connectivity_img_coh(time_series_original = eeg, label = f'EEG theta band (4-8)Hz', lowcut=4, highcut=8, real_patient=True, patient=i+1)
        functional_connectivity_img_coh(time_series_original = eeg, label = f'EEG delta band (0.5-4)Hz', lowcut=0.5, highcut=4, real_patient=True, patient=i+1)
        psd(eeg, label=f"EEG", real_patient=True, patient=i+1)

analyze_patients(paths=paths, prot_lev=prot_lev)
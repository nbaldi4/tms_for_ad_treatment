import pandas as pd
from pathlib import Path
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import global_variable_creation as gv
import time

from joblib import Parallel, delayed
from tqdm import tqdm
from data_analysis_utils import real_vs_stages_PSD, real_vs_stages_FC, to_common_indices, macro_fc, analyze_regular_data, analyze_reduced_data
import mne
from utils import psd, functional_connectivity_img_coh
from staging_highlights import collect_files
from additional_utils import extract_common_names

from pymatreader import read_mat
import mne

start = time.time()

# Rimuove il limite di larghezza della colonna (None = nessun limite)

root_folder_global_save = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\MCI_data"
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
                  path_20])#, path_21, path_22, path_23, path_24, path_25, path_26, path_27, path_28,
                  #path_29, path_30, path_31, path_32, path_33, path_34, path_35, path_36, path_37,
                  #path_38, path_39, path_40, path_41, path_42, path_43, path_44, path_45])

abeta_level = np.loadtxt(r"C:\Users\User\OneDrive - University of Pisa\Desktop\CSF_abeta.csv", skiprows=1)
tau_level = np.loadtxt(r"C:\Users\User\OneDrive - University of Pisa\Desktop\CSF_p_tau.csv", skiprows=1)
prot_lev = np.column_stack((abeta_level, tau_level))
gv.save_plots = True

def fun(path_ec):

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
    eeg_reshape = eeg_data.T[10000:20000, np.newaxis, :, np.newaxis] ########################################################

    # sample frequency
    gv.temp_avg_period = 1000 / raw_ec.info['sfreq']
    return eeg_reshape


def analyze_patients(paths):
    for i in range(len(paths)):
        p_id = f"{i+1:02d}"
        eeg = fun(paths[i])
        functional_connectivity_img_coh(time_series_original = eeg, label = f'EEG broad band (0.5-45)Hz', lowcut=0.5, highcut=45, real_patient=True, patient=p_id)
        functional_connectivity_img_coh(time_series_original = eeg, label = f'EEG beta band (13-30)Hz', lowcut=13, highcut=30, real_patient=True, patient=p_id)
        functional_connectivity_img_coh(time_series_original = eeg, label = f'EEG alpha band (8-13)Hz', lowcut=8, highcut=13, real_patient=True, patient=p_id)
        functional_connectivity_img_coh(time_series_original = eeg, label = f'EEG theta band (4-8)Hz', lowcut=4, highcut=8, real_patient=True, patient=p_id)
        functional_connectivity_img_coh(time_series_original = eeg, label = f'EEG delta band (0.5-4)Hz', lowcut=0.5, highcut=4, real_patient=True, patient=p_id)
        psd(eeg, label=f"EEG", real_patient=True, patient=p_id)

def process_single_patient(path, p_id):
    """Questa funzione contiene tutto il lavoro pesante per un singolo paziente."""
    # Caricamento dati
    eeg = fun(path)
    
    # Definizione delle bande per rendere il codice più pulito
    bands = [
        ('broad band', 0.5, 45),
        ('beta band', 13, 30),
        ('alpha band', 8, 13),
        ('theta band', 4, 8),
        ('delta band', 0.5, 4)
    ]
    
    # Calcolo Coerenza (FC) per ogni banda
    for name, low, high in bands:
        functional_connectivity_img_coh(
            time_series_original=eeg, 
            label=f'EEG {name} ({low}-{high})Hz', 
            lowcut=low, highcut=high, 
            real_patient=True, patient=p_id
        )
    
    # Calcolo PSD
    psd(eeg, label="EEG", real_patient=True, patient=p_id)
    return f"Paziente {p_id} processato."

def analyze_patients_parallel(paths, n_jobs=-3):
    """Versione parallelizzata della tua funzione originale."""
    # Prepariamo la lista dei task
    tasks = [
        (paths[i], f"{i+1:02d}") 
        for i in range(len(paths))
    ]
    
    # Lancio del calcolo parallelo
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_patient)(p, i) for p, i in tqdm(tasks, desc="Analisi Pazienti")
    )
    return results

def natural_keys(text):
    # Converte i numeri in interi per l'ordinamento corretto (1, 2, 10 invece di 1, 10, 2)
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(text))]

def apply_natural_keys(df):
    if not df.empty:
        # Estraiamo solo il nome del file dal path per l'ordinamento
        df['filename_only'] = df['file_path'].apply(lambda x: os.path.basename(x))
        df['sort_key'] = df['filename_only'].apply(natural_keys)
        df = df.sort_values('sort_key').drop(columns=['sort_key', 'filename_only'])
        df = df.reset_index(drop=True) # FONDAMENTALE per usare iloc correttamente
    return df

def file_extraction(files_group, p_idx):
    file_path = files_group.iloc[p_idx]['file_path']
    return np.load(file_path)

def build_patient_highlights_file(root_folder=r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data"):
    # 1. Raccolta file con i nuovi pattern dei pazienti
    alpha_band_files = collect_files(root_folder, files_name="FC_8-13_Hz_patient_*")
    beta_band_files = collect_files(root_folder, files_name="FC_13-30_Hz_patient_*")
    delta_band_files = collect_files(root_folder, files_name="FC_0.5-4_Hz_patient_*")
    theta_band_files = collect_files(root_folder, files_name="FC_4-8_Hz_patient_*")
    broad_band_files = collect_files(root_folder, files_name="FC_0.5-45_Hz_patient_*")
    PSD_files = collect_files(root_folder, files_name="PSD_EEG_patient_*")

    alpha_band_files = apply_natural_keys(alpha_band_files)
    beta_band_files = apply_natural_keys(beta_band_files)
    delta_band_files = apply_natural_keys(delta_band_files)
    theta_band_files = apply_natural_keys(theta_band_files)
    broad_band_files = apply_natural_keys(broad_band_files)
    PSD_files = apply_natural_keys(PSD_files)

    bands_files = [broad_band_files, delta_band_files, theta_band_files, alpha_band_files, beta_band_files]
    
    # Determiniamo il numero di pazienti (usando alpha come riferimento)
    num_patients = alpha_band_files.shape[0]
    if num_patients == 0:
        print("Nessun paziente trovato!")
        return None
    else:
        print(f"{num_patients} pazienti trovati")
    # 2. Inizializzazione contenitori (N_pazienti invece di 6x6)
    # Usiamo la stessa struttura di stage_metrics ma con la prima dimensione = num_patients
    psd_means = np.zeros((num_patients, 5)) 
    psd_max = np.zeros((num_patients, 5))
    freq_max = np.zeros(num_patients)
    psd_metrics = np.zeros((num_patients, 2))
    fc_reg_means = np.zeros((num_patients, 5, 2))
    fc_red_means = np.zeros((num_patients, 5, 2))
    fc_metrics = np.zeros((num_patients, 3))

    # Nodi critici per la sottomatrice 7x7 (se ti serve ancora)
    interested_nodes = [6, 7, 8, 9, 12, 13, 16]
    fc_7x7_means = np.zeros((num_patients, 5, 2))

    # 3. Estrazione dati per ogni paziente
    for p_idx in range(num_patients):
        # Nota: assumiamo che collect_files abbia ordinato i pazienti nello stesso modo per ogni banda
        
        # Estrazione FC per le 5 bande
        for b_idx in range(len(bands_files)):
            fc_matrix = file_extraction(bands_files[b_idx], p_idx)

            # Medie Regular e Reduced
            fc_reg_means[p_idx, b_idx, 0] = np.mean(fc_matrix['normal_size'])
            fc_reg_means[p_idx, b_idx, 1] = np.mean(np.abs(fc_matrix['normal_size']))
            fc_red_means[p_idx, b_idx, 0] = np.mean(fc_matrix['reduced_size'])
            fc_red_means[p_idx, b_idx, 1] = np.mean(np.abs(fc_matrix['reduced_size']))
            
            # Sottomatrice 7x7
            fc_7x7 = fc_matrix['normal_size'][np.ix_(interested_nodes, interested_nodes)]
            fc_7x7_means[p_idx, b_idx, 0] = np.mean(fc_7x7)
            fc_7x7_means[p_idx, b_idx, 1] = np.mean(np.abs(fc_7x7))

        # Estrazione PSD
        psd_path = PSD_files.iloc[p_idx]['file_path']
        psd_data = np.load(psd_path)
        psd = psd_data['psd_mean']
        freqs = psd_data['freqs']

        # Calcolo quantità PSD (somme e max)
        masks = [
            (freqs >= 0.5),                      # Broad
            (freqs < 4),                         # Delta
            (freqs >= 4) & (freqs < 8),          # Theta
            (freqs >= 8) & (freqs < 12),         # Alpha
            (freqs >= 12) & (freqs < 30)         # Beta
        ]

        for m_idx, mask in enumerate(masks):
            psd_means[p_idx, m_idx] = np.sum(psd[mask])
            psd_max[p_idx, m_idx] = np.max(psd[mask])

        # Metriche PSD (Rapporti)
        psd_metrics[p_idx, 0] = psd_means[p_idx, 3] / psd_means[p_idx, 0] # Alpha/Broad
        psd_metrics[p_idx, 1] = psd_means[p_idx, 1] / psd_means[p_idx, 0] # Delta/Broad
        psd_metrics[p_idx, 2] = psd_means[p_idx, 2] / psd_means[p_idx, 0] # Theta/Broad
        psd_metrics[p_idx, 3] = psd_means[p_idx, 4] / psd_means[p_idx, 0] # Beta/Broad

        # Frequenza di picco globale
        freq_max[p_idx] = freqs[np.argmax(psd)]

    # Metriche FC finali (Delta, Theta, Alpha abs mean)
    fc_metrics[:, 0] = fc_reg_means[:, 1, 1] # Delta
    fc_metrics[:, 1] = fc_reg_means[:, 2, 1] # Theta
    fc_metrics[:, 2] = fc_reg_means[:, 3, 1] # Alpha
    fc_metrics[:, 3] = fc_reg_means[:, 4, 1] # Beta
    fc_metrics[:, 4] = fc_reg_means[:, 0, 1] # Broad

    os.makedirs(root_folder_global_save, exist_ok=True)
    output = os.path.join(root_folder_global_save, f"patients_metrics.npz")
    np.savez_compressed(
            output,
            patient_id=np.linspace(1, num_patients, num_patients),
            fc_reg_means=fc_reg_means,
            fc_red_means=fc_red_means,
            psd_max=psd_max,
            freq_max=freq_max,
            psd_metrics=psd_metrics,
            fc_metrics=fc_metrics
        )

def print_patient_highlights(patients_metrics_path_file = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\MCI_data\patients_metrics.npz"):
    
    data = np.load(patients_metrics_path_file)
    p_max = data['psd_max']
    fc_means = data['fc_reg_means']

    bands = ['Broad', 'Delta', 'Theta', 'Alpha', 'Beta']

    print("-" * 60)
    print(f"{'PATIENT':<10} | {'BAND':<8} | {'PSD MAX':<12} | {'FC MEAN (abs)'}")
    print("-" * 60)

    num_pazienti = p_max.shape[0]

    for p in range(num_pazienti):
        for b_idx, band_name in enumerate(bands):
            psd_val = p_max[p, b_idx]
            fc_val = fc_means[p, b_idx, 1] #index 1 stands for absolute mean value

            p_label = f"Paz {p+1}" if b_idx == 0 else ""
            
            print(f"{p_label:<10} | {band_name:<8} | {psd_val:<12.4e} | {fc_val:.4f}")
        print("-" * 60)

def build_reduced_patient_highlights_file(root_folder=r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data", 
                                          fc_reference_path=r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Stages_24_01\tau0_abeta0_a0.0728_b0.0629_g0.1000_v10.37_n0.1448_simtime20000\FC_EEG alpha (8-13)Hz_tau_0_abeta_0_8-13_Hz.npz"):
    # 1. Raccolta file con i nuovi pattern dei pazienti
    alpha_band_files = collect_files(root_folder, files_name="FC_8-13_Hz_patient_*")
    beta_band_files = collect_files(root_folder, files_name="FC_13-30_Hz_patient_*")
    delta_band_files = collect_files(root_folder, files_name="FC_0.5-4_Hz_patient_*")
    theta_band_files = collect_files(root_folder, files_name="FC_4-8_Hz_patient_*")
    broad_band_files = collect_files(root_folder, files_name="FC_0.5-45_Hz_patient_*")
    PSD_files = collect_files(root_folder, files_name="PSD_EEG_patient_*")

    alpha_band_files = apply_natural_keys(alpha_band_files)
    beta_band_files = apply_natural_keys(beta_band_files)
    delta_band_files = apply_natural_keys(delta_band_files)
    theta_band_files = apply_natural_keys(theta_band_files)
    broad_band_files = apply_natural_keys(broad_band_files)
    PSD_files = apply_natural_keys(PSD_files)

    bands_files = [broad_band_files, delta_band_files, theta_band_files, alpha_band_files, beta_band_files]

    #for i, bf in enumerate(bands_files):
    #    print(f"Banda {i}: {bf.shape[0]} file")

    # Determiniamo il numero di pazienti (usando alpha come riferimento)
    num_patients = alpha_band_files.shape[0]
    if num_patients == 0:
        print("Nessun paziente trovato!")
        return None
    else:
        print(f"{num_patients} pazienti trovati")
    # 2. Inizializzazione contenitori (N_pazienti invece di 6x6)
    # Usiamo la stessa struttura di stage_metrics ma con la prima dimensione = num_patients
    psd_means = np.zeros((num_patients, 5)) 
    psd_max = np.zeros((num_patients, 5))
    psd_all_ch = np.zeros((num_patients, 52, 1025))
    freq_max = np.zeros(num_patients)
    psd_metrics = np.zeros((num_patients, 4))
    fc_reg_means = np.zeros((num_patients, 5, 2))
    fc_red_means = np.zeros((num_patients, 5, 2))
    fc_metrics = np.zeros((num_patients, 5))

    # Nodi critici per la sottomatrice 7x7 (se ti serve ancora)
    interested_nodes = [6, 7, 8, 9, 12, 13, 16]
    fc_7x7_means = np.zeros((num_patients, 5, 2))

    reference_matrices=np.load(fc_reference_path)
    fc_reference_matrix = reference_matrices['normal_size']

    electrodes_name_path = 'C:/Users/User/OneDrive - University of Pisa/Desktop/TVB_tutorials/Dati_Healthy/'

    # Channel names for real EEG (62 channels)
    eeg_real_data_names = np.load(electrodes_name_path + 'names_ch.npy')
    eeg_real_data_names = np.array(eeg_real_data_names)

    # Channel names for Brainstorm EEG (65 channels)
    eeg_brainstorm_65_names = np.array([
        'Fp1', 'Fp2', 'F4', 'F3', 'C3', 'C4', 'P4', 'P3', 'O2', 'O1',
        'F8', 'F7', 'T4', 'T3', 'T6', 'T5', 'Pz', 'Fz', 'IO1', 'IO2',
        'AF9', 'AF10', 'F9', 'F10', 'CB1', 'CB2', 'TP7', 'TP9', 'TP10',
        'TP8', 'Oz', 'Iz', 'PO4', 'PO3', 'CP5', 'CP6', 'CP1', 'CP2',
        'FT9', 'FT10', 'FC2', 'FC1', 'AF3', 'AF4', 'FC6', 'FC5', 'CPz',
        'P1', 'POz', 'P2', 'P6', 'C6', 'P5', 'C1', 'C2', 'C5', 'F2',
        'F6', 'F1', 'AF8', 'F5', 'AF7', 'Fpz', 'FCz', 'Cz', 'T8', 'T7',
        'P8', 'P7'
    ])

    common_names = extract_common_names()
    #print("Numero canali comuni:", len(common_names))
    #print(common_names)
    #print(len(common_names) == len(set(common_names)))

    eeg_real_data_names = np.load('C:/Users/User/OneDrive - University of Pisa/Desktop/TVB_tutorials/Dati_Healthy/' + 'names_ch.npy')

    indices_65 = [np.where(eeg_brainstorm_65_names == name)[0][0] for name in common_names]
    indices_62 = [np.where(eeg_real_data_names == name)[0][0] for name in common_names]

    missing_in_real = set(eeg_brainstorm_65_names) - set(eeg_real_data_names)
    missing_in_brainstorm = set(eeg_real_data_names) - set(eeg_brainstorm_65_names)

    #print("Mancanti nel reale:", missing_in_real)
    #print("Mancanti in Brainstorm:", missing_in_brainstorm)

    assert np.all(eeg_brainstorm_65_names[indices_65] == np.array(common_names))
    assert np.all(eeg_real_data_names[indices_62] == np.array(common_names))

    #for i in range(10):
    #    print(
    #        common_names[i],
    #        eeg_brainstorm_65_names[indices_65[i]],
    #        eeg_real_data_names[indices_62[i]]
    #    )

    # 3. Estrazione dati per ogni paziente
    for p_idx in range(num_patients):
        # Nota: assumiamo che collect_files abbia ordinato i pazienti nello stesso modo per ogni banda
        # Estrazione FC per le 5 bande
        for b_idx in range(len(bands_files)):
            fc_matrix = file_extraction(bands_files[b_idx], p_idx)

            real_matrix_sub = fc_matrix['normal_size'][np.ix_(indices_62, indices_62)]

            # Medie Regular e Reduced
            fc_reg_means[p_idx, b_idx, 0] = np.mean(real_matrix_sub)
            fc_reg_means[p_idx, b_idx, 1] = np.mean(np.abs(real_matrix_sub))
            fc_red_means[p_idx, b_idx, 0] = np.mean(fc_matrix['reduced_size'])
            fc_red_means[p_idx, b_idx, 1] = np.mean(np.abs(fc_matrix['reduced_size']))
            
            # Sottomatrice 7x7
            fc_7x7 = fc_matrix['normal_size'][np.ix_(interested_nodes, interested_nodes)]
            fc_7x7_means[p_idx, b_idx, 0] = np.mean(fc_7x7)
            fc_7x7_means[p_idx, b_idx, 1] = np.mean(np.abs(fc_7x7))

        # Estrazione PSD
        psd_path = PSD_files.iloc[p_idx]['file_path']
        psd_data = np.load(psd_path)
        freqs = psd_data['freqs']
        psd_all = psd_data['psd_all_channels']
        assert fc_matrix['normal_size'].shape[0] >= max(indices_62) + 1
        assert psd_all.shape[0] >= max(indices_62) + 1
        psd_all_ch = psd_all[indices_62, :]
        psd = np.mean(psd_all_ch, axis=0)
        # Calcolo quantità PSD (somme e max)
        masks = [
            (freqs >= 0.5),                      # Broad
            (freqs < 4),                         # Delta
            (freqs >= 4) & (freqs < 8),          # Theta
            (freqs >= 8) & (freqs < 12),         # Alpha
            (freqs >= 12) & (freqs < 30)         # Beta
        ]
        if p_idx == 0:
            i = 10
            check_channels = ['Fp1', 'Cz', 'Pz', 'O1', 'T5']

            for ch in check_channels:
                i = common_names.index(ch)
                print(
                    f"{ch:>3} | "
                    f"FC mean: {np.mean(real_matrix_sub[i]):.4e} | "
                    f"PSD mean: {np.mean(psd_all[i]):.4e}"
                )
            i = common_names.index('Cz')
            #print("Self:", real_matrix_sub[i, i])
            #print("Others:", np.mean(np.delete(real_matrix_sub[i], i)))

            #for band, mask in zip(
            #        ['delta','theta','alpha','beta'],
            #        [(freqs<4),(freqs>=4)&(freqs<8),(freqs>=8)&(freqs<13),(freqs>=13)&(freqs<30)]
            #):
            #    print(band, np.mean(psd[mask]))

                    
        for m_idx, mask in enumerate(masks):
            psd_means[p_idx, m_idx] = np.sum(psd[mask])
            psd_max[p_idx, m_idx] = np.max(psd[mask])

        # Metriche PSD (Rapporti)
        psd_metrics[p_idx, 0] = psd_means[p_idx, 1] / psd_means[p_idx, 0] # Delta/Broad
        psd_metrics[p_idx, 1] = psd_means[p_idx, 2] / psd_means[p_idx, 0] # Theta/Broad
        psd_metrics[p_idx, 2] = psd_means[p_idx, 3] / psd_means[p_idx, 0] # Alpha/Broad
        psd_metrics[p_idx, 3] = psd_means[p_idx, 4] / psd_means[p_idx, 0] # Beta/Broad

        # Frequenza di picco globale
        freq_max[p_idx] = freqs[np.argmax(psd)]

    # Metriche FC finali (Delta, Theta, Alpha abs mean)
    fc_metrics[:, 0] = fc_reg_means[:, 0, 1] # Broad
    fc_metrics[:, 1] = fc_reg_means[:, 1, 1] # Delta
    fc_metrics[:, 2] = fc_reg_means[:, 2, 1] # Theta
    fc_metrics[:, 3] = fc_reg_means[:, 3, 1] # Alpha
    fc_metrics[:, 4] = fc_reg_means[:, 4, 1] # Beta

    #print(fc_reg_means[:, 1, 1])

    os.makedirs(root_folder_global_save, exist_ok=True)
    output = os.path.join(root_folder_global_save, f"patients_common_indices_metrics.npz")
    np.savez_compressed(
            output,
            patient_id=np.array(np.linspace(1, num_patients, num_patients)),
            fc_reg_means=fc_reg_means,
            fc_red_means=fc_red_means,
            psd_all_ch=psd_all_ch,
            psd_max=psd_max,
            freq_max=freq_max,
            psd_metrics=psd_metrics,
            fc_metrics=fc_metrics
        )

build_reduced_patient_highlights_file()
#analyze_patients_parallel(paths=paths)
#analyze_patients(paths=paths)
#build_reduced_patient_highlights_file()

data = np.load(os.path.join(root_folder_global_save, f"patients_common_indices_metrics.npz"))

import os
import glob
import re
import numpy as np

base_path = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data"

# Trova tutte le cartelle patient_*
patient_dirs = glob.glob(os.path.join(base_path, "patient_*"))

data_loaded = {}
patients_number = 50
fc_reg = np.zeros((patients_number, 5, 2)) #patient_id, band [broad, delta, theta, alpha, beta], mean or mean absolute value
fc_red = np.zeros((patients_number, 5, 2))
psd_means = np.zeros((patients_number, 5)) #patient_id, band

for patient_dir in patient_dirs:
    patient_id = int(re.search(r"patient_(\d+)", patient_dir).group(1))
    npz_files = glob.glob(
        os.path.join(patient_dir, "PSD_EEG_patient_*.npz")
    )
    
    for npz_file in npz_files:
        #print(f"Apro: {npz_file}")
        psd_data = np.load(npz_file)
        
        # Salva il contenuto (opzionale)
        #data_loaded[npz_file] = dict(data)
        
        # Per vedere cosa contiene il file
        #print("Chiavi presenti:", data.files)
        psd = psd_data['psd_mean']
        freqs = psd_data['freqs']
        psd_means[patient_id, 0] = np.mean(psd)                               #broad
        psd_means[patient_id, 1] = np.mean(psd[(freqs < 4)])                  #delta
        psd_means[patient_id, 2] = np.mean(psd[(freqs >= 4) & (freqs < 8)])   #theta
        psd_means[patient_id, 3] = np.mean(psd[(freqs >= 8) & (freqs < 12)])  #alpha
        psd_means[patient_id, 4] = np.mean(psd[(freqs >= 12) & (freqs < 30)]) #beta

        # Cerca i file FC_8-12_Hz_patient_*.npz
    patient_id = int(re.search(r"patient_(\d+)", patient_dir).group(1))
    #print(patient_id)

    npz_files = glob.glob(
        os.path.join(patient_dir, "FC_0.5-45_Hz_patient_*.npz")
    )
    
    for npz_file in npz_files:
        #print(f"Apro: {npz_file}")
        fc_data = np.load(npz_file)
        
        # Salva il contenuto (opzionale)
        #data_loaded[npz_file] = dict(data)

        # Se vuoi accedere a una variabile specifica:
        fc_reg[patient_id, 0, 0] = np.mean(fc_data["normal_size"])
        fc_reg[patient_id, 0, 1] = np.mean(np.abs(fc_data["normal_size"]))
        fc_red[patient_id, 0, 0] = np.mean(fc_data["reduced_size"])
        fc_red[patient_id, 0, 1] = np.mean(np.abs(fc_data["reduced_size"]))

    npz_files = glob.glob(
        os.path.join(patient_dir, "FC_0.5-4_Hz_patient_*.npz")
    )
    
    for npz_file in npz_files:
        #print(f"Apro: {npz_file}")
        fc_data = np.load(npz_file)
        
        # Salva il contenuto (opzionale)
        #data_loaded[npz_file] = dict(data)

        # Se vuoi accedere a una variabile specifica:
        fc_reg[patient_id, 1, 0] = np.mean(fc_data["normal_size"])
        fc_reg[patient_id, 1, 1] = np.mean(np.abs(fc_data["normal_size"]))
        fc_red[patient_id, 1, 0] = np.mean(fc_data["reduced_size"])
        fc_red[patient_id, 1, 1] = np.mean(np.abs(fc_data["reduced_size"]))

    npz_files = glob.glob(
        os.path.join(patient_dir, "FC_4-8_Hz_patient_*.npz")
    )
    
    for npz_file in npz_files:
        #print(f"Apro: {npz_file}")
        fc_data = np.load(npz_file)
        
        # Salva il contenuto (opzionale)
        #data_loaded[npz_file] = dict(data)

        # Se vuoi accedere a una variabile specifica:
        fc_reg[patient_id, 2, 0] = np.mean(fc_data["normal_size"])
        fc_reg[patient_id, 2, 1] = np.mean(np.abs(fc_data["normal_size"]))
        fc_red[patient_id, 2, 0] = np.mean(fc_data["reduced_size"])
        fc_red[patient_id, 2, 1] = np.mean(np.abs(fc_data["reduced_size"]))

    npz_files = glob.glob(
        os.path.join(patient_dir, "FC_8-12_Hz_patient_*.npz")
    )
    
    for npz_file in npz_files:
        #print(f"Apro: {npz_file}")
        fc_data = np.load(npz_file)
        
        # Salva il contenuto (opzionale)
        #data_loaded[npz_file] = dict(data)

        # Se vuoi accedere a una variabile specifica:
        fc_reg[patient_id, 3, 0] = np.mean(fc_data["normal_size"])
        fc_reg[patient_id, 3, 1] = np.mean(np.abs(fc_data["normal_size"]))
        fc_red[patient_id, 3, 0] = np.mean(fc_data["reduced_size"])
        fc_red[patient_id, 3, 1] = np.mean(np.abs(fc_data["reduced_size"]))

    npz_files = glob.glob(
        os.path.join(patient_dir, "FC_12-30_Hz_patient_*.npz")
    )
    
    for npz_file in npz_files:
        #print(f"Apro: {npz_file}")
        fc_data = np.load(npz_file)
        
        # Salva il contenuto (opzionale)
        #data_loaded[npz_file] = dict(data)

        # Se vuoi accedere a una variabile specifica:
        fc_reg[patient_id, 4, 0] = np.mean(fc_data["normal_size"])
        fc_reg[patient_id, 4, 1] = np.mean(np.abs(fc_data["normal_size"]))
        fc_red[patient_id, 4, 0] = np.mean(fc_data["reduced_size"])
        fc_red[patient_id, 4, 1] = np.mean(np.abs(fc_data["reduced_size"]))

patients_metrics = [fc_reg, fc_red, psd_means]
stages = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Stages_21_01_2026_img_coh\stage_metrics.npz")
#print("Chiavi:", stages.files)
stages_fc_reg_means = stages['fc_reg_means'] #6x6x5x2
stages_fc_red_means = stages['fc_red_means'] #6x6x5x2
stages_psd_means = stages['psd_means'] # 6x6x5
freqs = stages['freqs']

metrics = [stages_fc_reg_means, stages_fc_red_means, stages_psd_means]

bands_number = 5
metrics_number = 5
tau_stages = 6
abeta_stages = 6
distances = np.zeros((patients_number, bands_number, metrics_number, tau_stages, abeta_stages))
reduced_distances = np.zeros((patients_number, tau_stages, abeta_stages))
# patients x bands [broad, delta, theta, alpha, beta] x metrics number [fc_reg, fc_reg_abs, fc_red, fc_red_abs] x tau_stages [6] x abeta_stages [6]

for pat in range(patients_number):
    for abeta in range(abeta_stages):
        for tau in range(tau_stages):
            for band in range(bands_number):
                distances[pat, band, 0, tau, abeta] = fc_reg[pat, band, 0] - stages_fc_reg_means[tau, abeta, band, 0]
                distances[pat, band, 1, tau, abeta] = fc_reg[pat, band, 1] - stages_fc_reg_means[tau, abeta, band, 1]
                distances[pat, band, 2, tau, abeta] = fc_red[pat, band, 0] - stages_fc_red_means[tau, abeta, band, 0]
                distances[pat, band, 3, tau, abeta] = fc_red[pat, band, 1] - stages_fc_red_means[tau, abeta, band, 1]
                distances[pat, band, 4, tau, abeta] = psd_means[pat, band] - stages_psd_means[tau, abeta, band]
            reduced_distances[pat, tau, abeta] = np.abs(distances[pat, band, 0, tau, abeta]) + np.abs(distances[pat, band, 1, tau, abeta]) + np.abs(distances[pat, band, 2, tau, abeta]) + np.abs(distances[pat, band, 3, tau, abeta]) + np.abs(distances[pat, band, 4, tau, abeta])

#print(reduced_distances)

# 1. Trova l'indice "piatto"
idx_min = np.argmin(reduced_distances[1,:,:])

# 2. Convertilo in coordinate (riga, colonna)
riga_min, colonna_min = np.unravel_index(idx_min, reduced_distances[1,:,:].shape)

idx_max = np.argmax(reduced_distances[1,:,:])

# 2. Convertilo in coordinate (riga, colonna)
riga_min, colonna_min = np.unravel_index(idx_min, reduced_distances[1,:,:].shape)

stages_sum = np.sum(reduced_distances[1,:,:])

stage_prob = -(reduced_distances[1,:,:]-np.argmax(reduced_distances[1,:,:]))
stage_prob /= np.sum(stage_prob)

print(stage_prob)

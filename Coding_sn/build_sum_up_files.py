import pandas as pd
from pathlib import Path
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import global_variable_creation as gv
import sns

from data_analysis_utils import real_vs_stages_PSD, real_vs_stages_FC, to_common_indices, macro_fc
from additional_utils import extract_common_names
from sklearn.feature_selection import mutual_info_regression
from specparam import SpectralGroupModel # Usiamo Group per matrici [n_regions, n_freqs]

import re
import pandas as pd
from pathlib import Path

def collect_files(root_folder_path, files_name="*.npz"):
    root_folder = Path(root_folder_path)
    data = []

    # Regex aggiornata: cerca tau e abeta seguiti da numeri
    # Cerca anche 'rs' (random seed) se presente nel percorso
    pattern = re.compile(r"tau_?(\d+).*?abeta_?(\d+)", re.IGNORECASE)
    rs_pattern = re.compile(r"_rs(\d+)", re.IGNORECASE)

    # rglob("*") cerca ricorsivamente. Usiamo il pattern passato come argomento.
    for file in root_folder.rglob(files_name):
        # Cerchiamo i valori sia nel nome del file che nel percorso (più sicuro)
        full_path_str = str(file)
        match = pattern.search(file.name)
        rs_match = rs_pattern.search(full_path_str)

        if match:
            tau_val = int(match.group(1))
            abeta_val = int(match.group(2))
            rs_val = int(rs_match.group(1)) if rs_match else 0

            data.append({
                "tau": tau_val,
                "abeta": abeta_val,
                "rs": rs_val,
                "file_name": file.name,
                "file_path": str(file.resolve())
            })

    return pd.DataFrame(data)

def file_extraction(files_group, tau_stage, abeta_stage, seed=5):
    # Filtriamo per tau e abeta
    mask = (files_group["tau"] == tau_stage) & (files_group["abeta"] == abeta_stage)
    
    # Se specifichi un seed (es. rs0), filtra anche quello
    if seed is not None:
        mask &= (files_group["rs"] == seed)
        
    file_sel = files_group[mask]

    if file_sel.empty:
        print(f"⚠️ Nessun file trovato per Tau:{tau_stage}, Abeta:{abeta_stage}")
        return None

    # Prendiamo il primo disponibile
    file_path_string = file_sel.iloc[0]["file_path"]
    return np.load(file_path_string)


######################## SIMULATED STAGES ####################################
root = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Stages_increase_b"
df_broad = collect_files(root, files_name="FC_EEG_tau_*")
df_delta = collect_files(root, files_name="FC_EEG delta_tau_*")
df_theta = collect_files(root, files_name="FC_EEG theta_tau_*")
df_alpha = collect_files(root, files_name="FC_EEG alpha_tau_*")
df_beta = collect_files(root, files_name="FC_EEG beta_tau_*")
df_psd = collect_files(root, files_name="PSD_EEG_tau_*")
df_psd = df_psd[~df_psd["file_name"].str.contains("interested", case=False)]

stages = [0,1,2,3,4,5]
seeds = [0]#,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
bands = [0,1,2,3,4]
fc_matrices = np.zeros((len(stages), len(stages), len(bands), len(seeds)), dtype=object)
psd_arrays = np.zeros((len(stages), len(stages), len(seeds)), dtype=object)
psd_freqs = np.zeros((len(stages), len(stages), len(seeds)), dtype=object)
for abeta in stages:
    for tau in stages:
        for seed in seeds:
            data_broad = file_extraction(df_broad, tau_stage=tau, abeta_stage=abeta, seed=seed)
            data_delta = file_extraction(df_delta, tau_stage=tau, abeta_stage=abeta, seed=seed)
            data_theta = file_extraction(df_theta, tau_stage=tau, abeta_stage=abeta, seed=seed)
            data_alpha = file_extraction(df_alpha, tau_stage=tau, abeta_stage=abeta, seed=seed)
            data_beta = file_extraction(df_beta, tau_stage=tau, abeta_stage=abeta, seed=seed)
            data_psd = file_extraction(df_psd, tau_stage=tau, abeta_stage=abeta, seed=seed)
            datas = [data_broad, data_delta, data_theta, data_alpha, data_beta]
            
            for band in bands:
                if datas[band]:
                    reg_FC = datas[band]['normal_size']
                    fc_matrices[abeta, tau, band, seed] = reg_FC
                    mean = np.mean(reg_FC)
                    abs_mean = np.mean(np.abs(reg_FC))
                    datas[band].close()

            psd_all = data_psd['psd_all_channels']
            psd_arrays[abeta, tau, seed] = np.mean(psd_all, axis=0)
            freqs = data_psd['freqs']
            psd_freqs[abeta, tau, seed] = freqs
metadata_fc = {
    "stages": stages,
    "bands": ["broad", "delta", "theta", "alpha", "beta"],
    "seeds": seeds,
    "description": "Regular (65x65) size FC matrices. First index represent abeta stages, second tau stages, third bands, fouth seeds"
}

output_path_fc = os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Staging_highlights", "Stages_increase_b_simulated_stages_fc.npz")

np.savez_compressed(
    output_path_fc, 
    fc_matrices=fc_matrices, 
    metadata=metadata_fc
)

# conversione a numpy liscio
fc_matrices = np.stack(fc_matrices, axis=0) #numpy ndarray

# nuovo salvataggio 
#np.savez(os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Staging_highlights",'simulated_stages_fc_clean.npz'), 
#         fc_matrices=fc_matrices,
#         metadata_fc=metadata_fc
#)

metadata_psd = {
    "stages": stages,
    "seeds": seeds,
    "description": "PSD arrays (1x1025). First index represent abeta stages, second tau stages, third seeds. Frequencies are also present"
}

output_path = os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Staging_highlights", "Stages_increase_b_simulated_stages_psd.npz")

np.savez_compressed(
    output_path, 
    psd_arrays=psd_arrays,
    psd_freqs=psd_freqs,
    metadata=metadata_psd
)

# conversione a numpy liscio
psd_arrays = np.stack(psd_arrays, axis=0) #numpy ndarray
psd_freqs = np.stack(psd_freqs, axis=0)

# nuovo salvataggio 
#np.savez(os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Staging_highlights",'simulated_stages_psd_clean.npz'), 
#         psd_arrays=psd_arrays,
#         psd_freqs=psd_freqs,
#         metadata=metadata_psd
#)


loaded_psd_data = np.load(os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Staging_highlights", "Stages_increase_b_simulated_stages_psd.npz"), allow_pickle=True)
loaded_fc_data = np.load(os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Staging_highlights", "Stages_increase_b_simulated_stages_fc.npz"), allow_pickle=True)
fc_from_file = loaded_fc_data['fc_matrices']
psd_from_file = loaded_psd_data['psd_arrays']
print(fc_from_file.shape)
print(psd_from_file.shape)
print(fc_from_file[0,0,0,0].shape)
print(psd_from_file[0,0,0].shape)
# Verifichiamo prima la forma (se sono diverse, sono sicuramente diversi)
if fc_from_file.shape == fc_matrices.shape:
    # Appiattiamo entrambi e usiamo un ciclo generator per controllare cella per cella
    # Questo evita l'ambiguità perché np.array_equal viene chiamato sulle singole matrici 65x65
    is_identical = all(np.array_equal(a, b) for a, b in zip(fc_from_file.flat, fc_matrices.flat))
    
    if is_identical:
        print("✅ Identical FC")
    else:
        print("❌ FC differiscono nel contenuto")
else:
    print("❌ FC differiscono nella forma (Shape)")

# TEST PSD: Nota che la chiave è 'psd_arrays', non 'psd_matrices' 
# (controlla come l'hai salvata sopra!)
if psd_from_file.shape == psd_arrays.shape:
    # Appiattiamo entrambi e usiamo un ciclo generator per controllare cella per cella
    # Questo evita l'ambiguità perché np.array_equal viene chiamato sulle singole matrici 65x65
    is_identical = all(np.array_equal(a, b) for a, b in zip(psd_from_file.flat, psd_arrays.flat))
    
    if is_identical:
        print("✅ Identical PSD")
    else:
        print("❌ PSD differiscono nel contenuto")
else:
    print("❌ PSD differiscono nella forma (Shape)")

"""
fc_A = np.ones((2,2))
fc_A[0,0] = 5
fc_A[1,0] = 3

fc_B = np.ones((3,2))
fc_B[0,0] = 5
fc_B[1,0] = 4

if fc_A.shape == fc_B.shape:
    # Appiattiamo entrambi e usiamo un ciclo generator per controllare cella per cella
    # Questo evita l'ambiguità perché np.array_equal viene chiamato sulle singole matrici 65x65
    is_identical = all(np.array_equal(a, b) for a, b in zip(fc_A.flat, fc_B.flat))
    
    if is_identical:
        print("✅ Identical FC")
    else:
        print("❌ FC differiscono nel contenuto")
else:
    print("❌ FC differiscono nella forma (Shape)")

# TEST PSD: Nota che la chiave è 'psd_arrays', non 'psd_matrices' 
# (controlla come l'hai salvata sopra!)

psd_A = np.ones(3)
psd_A[0] = 1
psd_A[1] = 3

psd_B = np.ones(3)
psd_B[0] = 5
psd_B[1] = 3
if psd_A.shape == psd_B.shape:
    # Appiattiamo entrambi e usiamo un ciclo generator per controllare cella per cella
    # Questo evita l'ambiguità perché np.array_equal viene chiamato sulle singole matrici 65x65
    is_identical = all(np.array_equal(a, b) for a, b in zip(psd_A.flat, psd_B.flat))
    
    if is_identical:
        print("✅ Identical PSD")
    else:
        print("❌ PSD differiscono nel contenuto")
else:
    print("❌ PSD differiscono nella forma (Shape)")


"""
print('abeta 0, tau 4, banda theta, seed 3')
print(fc_matrices[0, 4, 2, 3].shape, '; ', np.mean(np.abs(fc_matrices[0, 4, 2, 3])))
print(psd_arrays[0, 4, 3].shape, '; ', np.max(psd_arrays[0, 4, 3]))
print(psd_freqs[0, 4, 3].shape)
print('abeta 1, tau 5, banda alpha, seed 12')
print(fc_matrices[1, 5, 3, 12].shape, '; ', np.mean(np.abs(fc_matrices[1, 5, 3, 12])))
print(psd_arrays[1, 5, 12].shape, '; ', np.max(psd_arrays[1, 5, 12]))
print(psd_freqs[1, 5, 12].shape)
print('abeta 2, tau 1, banda delta, seed 1')
print(fc_matrices[2, 1, 1, 1].shape, '; ', np.mean(np.abs(fc_matrices[2, 1, 1, 1])))
print(psd_arrays[2, 1, 1].shape, '; ', np.max(psd_arrays[2, 1, 1]))
print(psd_freqs[2, 1, 1].shape)


############################## HC PATIENTS ##########################################

fc_healty_broad = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\HC_data\fc_ctr_preview.npy")
fc_healty_delta = np.zeros(fc_healty_broad.shape)
fc_healty_theta = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\HC_data\fc_preview_ctr_theta.npy") #sono tutti nan
fc_healty_alpha = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\HC_data\fc_preview_ctr_alpha.npy")
fc_healty_beta = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\HC_data\fc_preview_ctr_beta.npy")
fc_healty_gamma = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\HC_data\fc_preview_ctr_gamma.npy")
psd_healty = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\HC_data\psd_ctr_preview.npy")
"""
hc_patients = len(fc_healty_broad[:,0,0])
hc_mean_abs_fc = np.zeros((hc_patients, 6))
hc_psd_rel_power = np.zeros((hc_patients, 5))
for i in range(hc_patients):
    hc_mean_abs_fc[i, 0] = np.mean(np.abs(fc_healty_broad[i,:,:]))
    hc_mean_abs_fc[i, 1] = np.mean(np.abs(fc_healty_delta[i,:,:]))
    hc_mean_abs_fc[i, 2] = 0#np.mean(np.abs(fc_healty_theta[i,:,:]))
    hc_mean_abs_fc[i, 3] = np.mean(np.abs(fc_healty_alpha[i,:,:]))
    hc_mean_abs_fc[i, 4] = np.mean(np.abs(fc_healty_beta[i,:,:]))
    hc_mean_abs_fc[i, 5] = np.mean(np.abs(fc_healty_gamma[i,:,:]))
    psd_healty_tot = psd_healty[i][1:].astype(float)
    freqs = np.linspace(0.25,256,len(psd_healty_tot))
    psd_broad = np.sum(psd_healty_tot)                               #broad
    psd_delta = np.sum(psd_healty_tot[(freqs < 4)])                  #delta
    psd_theta = np.sum(psd_healty_tot[(freqs >= 4) & (freqs < 8)])   #theta
    psd_alpha = np.sum(psd_healty_tot[(freqs >= 8) & (freqs < 13)])  #alpha
    psd_beta = np.sum(psd_healty_tot[(freqs >= 13) & (freqs < 30)])  #beta
    psd_gamma = np.sum(psd_healty_tot[(freqs >= 30) & (freqs < 45)]) #gamma
    hc_psd_rel_power[i, 0] = psd_delta/psd_broad
    hc_psd_rel_power[i, 1] = psd_theta/psd_broad
    hc_psd_rel_power[i, 2] = psd_alpha/psd_broad
    hc_psd_rel_power[i, 3] = psd_beta/psd_broad
    hc_psd_rel_power[i, 4] = psd_gamma/psd_broad

print(hc_psd_rel_power[0, :])
"""
# Definiamo il percorso di salvataggio
save_path = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Patients_highlights\hc_summary_stats.npz"
"""
# Salvataggio compresso
np.savez_compressed(
    save_path,
    hc_fc_abs_mean=hc_mean_abs_fc,       # Shape: (n_pazienti, 6)
    hc_psd_rel_pow=hc_psd_rel_power,    # Shape: (n_pazienti, 5)
    #freqs_axis=freqs,            # Salviamo anche l'asse delle frequenze per sicurezza
    metadata="HC patient statistics: hc_fc_abs_mean (n_patients, bands) with bands = [broad, delta, theta, alpha, beta, gamma] is the average of the absolute values of each fc; hc_psd_rel_pow (n_patients, bands) with bands = [delta, theta, alpha, beta, gamma] are relative power normalized with PSD broad_band, freqs_axis is an array with frequencies values"
)


loaded_data_hc = np.load(save_path, allow_pickle=True)
fc_hc = loaded_data_hc['hc_fc_abs_mean']
psd_hc = loaded_data_hc['hc_psd_rel_pow']
# Verifichiamo prima la forma (se sono diverse, sono sicuramente diversi)
if fc_hc.shape == hc_mean_abs_fc.shape:
    # Appiattiamo entrambi e usiamo un ciclo generator per controllare cella per cella
    # Questo evita l'ambiguità perché np.array_equal viene chiamato sulle singole matrici 65x65
    is_identical = all(np.array_equal(a, b) for a, b in zip(fc_hc.flat, hc_mean_abs_fc.flat))
    
    if is_identical:
        print("✅ Identical FC")
    else:
        print("❌ FC differiscono nel contenuto")
else:
    print("❌ FC differiscono nella forma (Shape)")

# TEST PSD: Nota che la chiave è 'psd_arrays', non 'psd_matrices' 
# (controlla come l'hai salvata sopra!)
if psd_hc.shape == hc_psd_rel_power.shape:
    # Appiattiamo entrambi e usiamo un ciclo generator per controllare cella per cella
    # Questo evita l'ambiguità perché np.array_equal viene chiamato sulle singole matrici 65x65
    is_identical = all(np.array_equal(a, b) for a, b in zip(psd_hc.flat, hc_psd_rel_power.flat))
    
    if is_identical:
        print("✅ Identical PSD")
    else:
        print("❌ PSD differiscono nel contenuto")
else:
    print("❌ PSD differiscono nella forma (Shape)")
"""
"""
print(np.mean(np.abs(fc_healty_broad[11, :, :])), ' ', hc_mean_abs_fc[11, 0])
hc_corr_viz = 2 * (fc_healty_broad[11, :, :] - np.min(fc_healty_broad[11, :, :])) / (np.max(fc_healty_broad[11, :, :]) - np.min(fc_healty_broad[11, :, :])) - 1
    # Plot corr
fig = plt.figure(figsize=(12, 8)) 
ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
im = ax.imshow(hc_corr_viz, cmap='bwr', vmin=-1, vmax=1, aspect='equal') 
cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
plt.colorbar(im, cax=cax)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(freqs, psd_healty[11][1:], linewidth=3)
plt.fill_between(freqs, psd_healty[11][1:], alpha=0.35)
plt.title('PSD', fontsize=20)
#plt.ylim(0, 1.01)
plt.xlim(0, 45)
plt.xlabel('Frequency [Hz]', fontsize=16)
plt.ylabel('PSD', fontsize=16)
plt.grid(True)
plt.show()
"""
############################### MCI PATIENTS ##############################

patient_data = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\MCI_data\patients_metrics.npz")
patient_psd_metrics = patient_data['psd_metrics'] #num_patients, metrics [delta/broad; theta/broad; alpha/broad; beta/broad]
"""
print(patient_psd_metrics.shape)
print('patient 11, delta/broad ', patient_psd_metrics[10, 0])
print('patient 11, theta/broad ', patient_psd_metrics[10, 1])
print('patient 11, alpha/broad ', patient_psd_metrics[10, 2])
print('patient 11, beta/broad ', patient_psd_metrics[10, 3])
print('patient 11, gamma/broad ', patient_psd_metrics[10, 4])
print('patient 126, delta/broad ', patient_psd_metrics[125, 0])
print('patient 126, theta/broad ', patient_psd_metrics[125, 1])
print('patient 126, alpha/broad ', patient_psd_metrics[125, 2])
print('patient 126, beta/broad ', patient_psd_metrics[125, 3])
print('patient 126, gamma/broad ', patient_psd_metrics[125, 4])
print('patient 34, delta/broad ', patient_psd_metrics[33, 0])
print('patient 34, theta/broad ', patient_psd_metrics[33, 1])
print('patient 34, alpha/broad ', patient_psd_metrics[33, 2])
print('patient 34, beta/broad ', patient_psd_metrics[33, 3])
print('patient 34, gamma/broad ', patient_psd_metrics[33, 4])
"""
patient_fc_metrics = patient_data['fc_metrics'] #num_patients, metrics [regular abs broad; regular abs delta; regular abs theta; regular abs alpha; rgular abs beta]
print(patient_fc_metrics.shape)
"""
print('patient 11, broad ', patient_fc_metrics[10, 0])
print('patient 2, delta ', patient_fc_metrics[1, 1])
print('patient 112, theta ', patient_fc_metrics[111, 2])
print('patient 26, alpha ', patient_fc_metrics[25, 3])
print('patient 26, beta ', patient_fc_metrics[25, 4])
"""
patient_metrics = np.zeros((len(patient_psd_metrics[:,0]), len(patient_psd_metrics[0,:]) + len(patient_fc_metrics[0,:])))
abeta_level = np.loadtxt(r"C:\Users\User\OneDrive - University of Pisa\Desktop\CSF_abeta.csv", skiprows=1)
tau_level = np.loadtxt(r"C:\Users\User\OneDrive - University of Pisa\Desktop\CSF_p_tau.csv", skiprows=1)

prot_lev = np.column_stack((abeta_level, tau_level))
#print(prot_lev)

mci_mean_abs_fc = np.zeros((59, 5))
mci_psd = np.zeros((59, 4))
mci_csf = np.zeros((59, 2))
i=0
for j in range(len(prot_lev)):
    if (prot_lev[j,0] != 0) & (prot_lev[j,1] != 0):
        if i == 58: print(j)
        mci_psd[i,0] = patient_psd_metrics[j,0] #delta/broad
        mci_psd[i,1] = patient_psd_metrics[j,1] #theta/broad
        mci_psd[i,2] = patient_psd_metrics[j,2] #alpha/broad
        mci_psd[i,3] = patient_psd_metrics[j,3] #beta/broad
        mci_mean_abs_fc[i,0] = patient_fc_metrics[j,0] #regular abs broad
        mci_mean_abs_fc[i,1] = patient_fc_metrics[j,1] #regular abs delta
        mci_mean_abs_fc[i,2] = patient_fc_metrics[j,2] #regular abs theta
        mci_mean_abs_fc[i,3] = patient_fc_metrics[j,3] #regular abs alpha
        mci_mean_abs_fc[i,4] = patient_fc_metrics[j,4] #regular abs beta
        mci_csf[i,:] = prot_lev[j,:]
        i += 1

save_path = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Patients_highlights\mci_summary_stats.npz"
"""
# Salvataggio compresso
np.savez_compressed(
    save_path,
    mci_fc_abs_mean=mci_mean_abs_fc,       # Shape: (n_pazienti, 5)
    mci_psd_rel_pow=mci_psd,    # Shape: (n_pazienti, 5)
    mci_csf=mci_csf,            # Salviamo anche l'asse delle frequenze per sicurezza
    metadata="HC patient statistics: mci_fc_abs_mean (n_patients, bands) with bands = [broad, delta, theta, alpha, beta] is the average of the absolute values of each fc; mci_psd_rel_pow is psd relative power corresponding to the follwing bands: [delta/broad, theta/broad, alpha/broad, beta/broad]; mci is an array num_patientsx2 where first column is abeta level and second column is tau level"
)
"""
mci_data = np.load(save_path)
psd = mci_data['mci_psd_rel_pow']
fc = mci_data['mci_fc_abs_mean']
csf = mci_data['mci_csf']

if mci_csf.shape == csf.shape:
    # Appiattiamo entrambi e usiamo un ciclo generator per controllare cella per cella
    # Questo evita l'ambiguità perché np.array_equal viene chiamato sulle singole matrici 65x65
    is_identical = all(np.array_equal(a, b) for a, b in zip(mci_csf.flat, csf.flat))
    
    if is_identical:
        print("✅ Identical FC")
    else:
        print("❌ FC differiscono nel contenuto")
else:
    print("❌ FC differiscono nella forma (Shape)")

# TEST PSD: Nota che la chiave è 'psd_arrays', non 'psd_matrices' 
# (controlla come l'hai salvata sopra!)
if psd.shape == mci_psd.shape:
    # Appiattiamo entrambi e usiamo un ciclo generator per controllare cella per cella
    # Questo evita l'ambiguità perché np.array_equal viene chiamato sulle singole matrici 65x65
    is_identical = all(np.array_equal(a, b) for a, b in zip(psd.flat, mci_psd.flat))
    
    if is_identical:
        print("✅ Identical PSD")
    else:
        print("❌ PSD differiscono nel contenuto")
else:
    print("❌ PSD differiscono nella forma (Shape)")

# (controlla come l'hai salvata sopra!)
if fc.shape == mci_mean_abs_fc.shape:
    # Appiattiamo entrambi e usiamo un ciclo generator per controllare cella per cella
    # Questo evita l'ambiguità perché np.array_equal viene chiamato sulle singole matrici 65x65
    is_identical = all(np.array_equal(a, b) for a, b in zip(fc.flat, mci_mean_abs_fc.flat))
    
    if is_identical:
        print("✅ Identical PSD")
    else:
        print("❌ PSD differiscono nel contenuto")
else:
    print("❌ PSD differiscono nella forma (Shape)")

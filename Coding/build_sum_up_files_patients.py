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
from specparam import SpectralGroupModel

import re
import pandas as pd
from pathlib import Path

############################## PATIENTS ##########################################
df = pd.read_csv(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\CSF_condition_id.csv")

#print(df.head())
patient_ids = df['num_DONGNOCCHI'].values
patient_cond = df['Conditions'].values
patient_abeta = df['ABeta'].values
patient_ptau = df['P-Tau'].values

df_hs = df[df['Conditions'] == 'HS'] 
hs_ids = df_hs['num_DONGNOCCHI'].values
hs_abeta = df_hs['ABeta'].values
hs_ptau = df_hs['P-Tau'].values

df_ad = df[(df['Conditions'] == 'SCD') | (df['Conditions'] == 'MCI')] 
ad_ids = df_ad['num_DONGNOCCHI'].values
ad_abeta = df_ad['ABeta'].values
ad_ptau = df_ad['P-Tau'].values
#print(ad_ids, ad_abeta, ad_ptau)

patient_data = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\MCI_data\patients_metrics_ref_freqs.npz")
patient_psd_metrics = patient_data['psd_metrics'] #num_patients, metrics [delta/broad; theta/broad; alpha/broad; beta/broad]
patient_fc_metrics = patient_data['fc_metrics'] #num_patients, metrics [regular abs broad; regular abs delta; regular abs theta; regular abs alpha; rgular abs beta]
#print(patient_fc_metrics.shape)

patient_metrics = np.zeros((len(patient_psd_metrics[:,0]), len(patient_psd_metrics[0,:]) + len(patient_fc_metrics[0,:])))

mci_mean_abs_fc = np.zeros((59, 5))
mci_psd = np.zeros((59, 4))
mci_csf = np.zeros((59, 2))

i = 0
for j in range(len(ad_abeta)):
    if not np.isnan(ad_abeta[j]) and not np.isnan(ad_ptau[j]):
        #print(ad_abeta[j], ad_ptau[j], ad_ids[j])
        mci_psd[i,0] = patient_psd_metrics[ad_ids[j]-1,0] #delta/broad
        mci_psd[i,1] = patient_psd_metrics[ad_ids[j]-1,1] #theta/broad
        mci_psd[i,2] = patient_psd_metrics[ad_ids[j]-1,2] #alpha/broad
        mci_psd[i,3] = patient_psd_metrics[ad_ids[j]-1,3] #beta/broad
        mci_mean_abs_fc[i,0] = patient_fc_metrics[ad_ids[j]-1,0] #regular abs broad
        mci_mean_abs_fc[i,1] = patient_fc_metrics[ad_ids[j]-1,1] #regular abs delta
        mci_mean_abs_fc[i,2] = patient_fc_metrics[ad_ids[j]-1,2] #regular abs theta
        mci_mean_abs_fc[i,3] = patient_fc_metrics[ad_ids[j]-1,3] #regular abs alpha
        mci_mean_abs_fc[i,4] = patient_fc_metrics[ad_ids[j]-1,4] #regular abs beta
        mci_csf[i,0] = ad_abeta[j]
        mci_csf[i,1] = ad_ptau[j]
        #if i <= 5:
            #print('paziente numero', ad_ids[j], ad_abeta[i], ad_ptau[i])
        i +=1

save_path = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Patients_highlights\mci_summary_stats_ref_freqs.npz"

np.savez_compressed(
    save_path,
    mci_fc_abs_mean=mci_mean_abs_fc,       # Shape: (n_pazienti, 5)
    mci_psd_rel_pow=mci_psd,    # Shape: (n_pazienti, 5)
    mci_csf=mci_csf,            # Salviamo anche l'asse delle frequenze per sicurezza
    metadata="HC patient statistics: mci_fc_abs_mean (n_patients, bands) with bands = [broad, delta, theta, alpha, beta] is the average of the absolute values of each fc; mci_psd_rel_pow is psd relative power corresponding to the follwing bands: [delta/broad, theta/broad, alpha/broad, beta/broad]; mci_csf is an array num_patientsx2 where first column is abeta level and second column is tau level"
)

mci_data = np.load(save_path)
mci_fc_abs_mean = mci_data['mci_fc_abs_mean']
mci_psd_rel_pow = mci_data['mci_psd_rel_pow']
mci_csf_values = mci_data['mci_csf']
print(mci_fc_abs_mean.shape)
print(mci_psd_rel_pow.shape)
print(mci_csf_values.shape)

# (controlla come l'hai salvata sopra!)
if mci_fc_abs_mean.shape == mci_mean_abs_fc.shape:
    # Appiattiamo entrambi e usiamo un ciclo generator per controllare cella per cella
    # Questo evita l'ambiguità perché np.array_equal viene chiamato sulle singole matrici 65x65
    is_identical = all(np.array_equal(a, b) for a, b in zip(mci_fc_abs_mean.flat, mci_mean_abs_fc.flat))
    
    if is_identical:
        print("✅ Identical FC")
    else:
        print("❌ FC differiscono nel contenuto")
else:
    print("❌ FC differiscono nella forma (Shape)")
print(mci_fc_abs_mean[:5])
print(mci_fc_abs_mean[-5:])

    # (controlla come l'hai salvata sopra!)
if mci_psd_rel_pow.shape == mci_psd.shape:
    # Appiattiamo entrambi e usiamo un ciclo generator per controllare cella per cella
    # Questo evita l'ambiguità perché np.array_equal viene chiamato sulle singole matrici 65x65
    is_identical = all(np.array_equal(a, b) for a, b in zip(mci_psd_rel_pow.flat, mci_psd.flat))
    
    if is_identical:
        print("✅ Identical PSD")
    else:
        print("❌ PSD differiscono nel contenuto")
else:
    print("❌ PSD differiscono nella forma (Shape)")
print(mci_psd_rel_pow[:5])
print(mci_psd_rel_pow[-5:])

    # (controlla come l'hai salvata sopra!)
if mci_csf_values.shape == mci_csf.shape:
    # Appiattiamo entrambi e usiamo un ciclo generator per controllare cella per cella
    # Questo evita l'ambiguità perché np.array_equal viene chiamato sulle singole matrici 65x65
    is_identical = all(np.array_equal(a, b) for a, b in zip(mci_csf_values.flat, mci_csf.flat))
    
    if is_identical:
        print("✅ Identical CSF")
    else:
        print("❌ CSF differiscono nel contenuto")
else:
    print("❌ CSF differiscono nella forma (Shape)")
print(mci_csf[:5])
print(mci_csf[-5:])

hc_mean_abs_fc = np.zeros((19, 5))
hc_psd = np.zeros((19, 4))

i = 0
for j in range(len(hs_ids)):
    hc_psd[i,0] = patient_psd_metrics[hs_ids[j]-1,0] #delta/broad
    hc_psd[i,1] = patient_psd_metrics[hs_ids[j]-1,1] #theta/broad
    hc_psd[i,2] = patient_psd_metrics[hs_ids[j]-1,2] #alpha/broad
    hc_psd[i,3] = patient_psd_metrics[hs_ids[j]-1,3] #beta/broad
    hc_mean_abs_fc[i,0] = patient_fc_metrics[hs_ids[j]-1,0] #regular abs broad
    hc_mean_abs_fc[i,1] = patient_fc_metrics[hs_ids[j]-1,1] #regular abs delta
    hc_mean_abs_fc[i,2] = patient_fc_metrics[hs_ids[j]-1,2] #regular abs theta
    hc_mean_abs_fc[i,3] = patient_fc_metrics[hs_ids[j]-1,3] #regular abs alpha
    hc_mean_abs_fc[i,4] = patient_fc_metrics[hs_ids[j]-1,4] #regular abs beta
    if i <= 5:
        print('paziente numero', hs_ids[j], hc_mean_abs_fc[i,0], hc_psd[i,0])
    i +=1

save_path = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Patients_highlights\hc_summary_stats_ref_freqs.npz"

np.savez_compressed(
    save_path,
    hc_fc_abs_mean=hc_mean_abs_fc,       # Shape: (n_pazienti, 6)
    hc_psd_rel_pow=hc_psd,    # Shape: (n_pazienti, 5)
    metadata="HC patient statistics: hc_fc_abs_mean (n_patients, bands) with bands = [broad, delta, theta, alpha, beta, gamma] is the average of the absolute values of each fc; hc_psd_rel_pow (n_patients, bands) with bands = [delta, theta, alpha, beta, gamma] are relative power normalized with PSD broad_band, freqs_axis is an array with frequencies values"
)

hc_data = np.load(save_path)
hc_fc_abs_mean = hc_data['hc_fc_abs_mean']
hc_psd_rel_pow = hc_data['hc_psd_rel_pow']
print(hc_fc_abs_mean.shape)
print(hc_psd_rel_pow.shape)

# (controlla come l'hai salvata sopra!)
if hc_fc_abs_mean.shape == hc_mean_abs_fc.shape:
    # Appiattiamo entrambi e usiamo un ciclo generator per controllare cella per cella
    # Questo evita l'ambiguità perché np.array_equal viene chiamato sulle singole matrici 65x65
    is_identical = all(np.array_equal(a, b) for a, b in zip(hc_fc_abs_mean.flat, hc_mean_abs_fc.flat))
    
    if is_identical:
        print("✅ Identical FC")
    else:
        print("❌ FC differiscono nel contenuto")
else:
    print("❌ FC differiscono nella forma (Shape)")
print(hc_fc_abs_mean)

    # (controlla come l'hai salvata sopra!)
if hc_psd_rel_pow.shape == hc_psd.shape:
    # Appiattiamo entrambi e usiamo un ciclo generator per controllare cella per cella
    # Questo evita l'ambiguità perché np.array_equal viene chiamato sulle singole matrici 65x65
    is_identical = all(np.array_equal(a, b) for a, b in zip(hc_psd_rel_pow.flat, hc_psd.flat))
    
    if is_identical:
        print("✅ Identical PSD")
    else:
        print("❌ PSD differiscono nel contenuto")
else:
    print("❌ PSD differiscono nella forma (Shape)")
print(hc_psd_rel_pow)
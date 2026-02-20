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

fc_mci_preview = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\MCI_data\fc_mci_preview.npy")
mci_preview_metrics = np.zeros((len(abeta_MCI_patients), 3))
mci_preview_metrics[:,0] = np.mean(fc_mci_preview, axis=(1,2))
psd_mci_preview = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\MCI_data\psd_mci_preview.npy")
psd_mci_preview_means = np.zeros((len(tau_MCI_patients), 2))
prot_lev_mci = np.column_stack((abeta_MCI_patients, tau_MCI_patients))
for i in range(len(tau_MCI_patients)):
    mci_preview_metrics[:,1] = np.sum(psd_mci_preview[i, 2:16])/np.sum(psd_mci_preview[i, 2:180])
    mci_preview_metrics[:,2] = np.sum(psd_mci_preview[i, 32:52])/np.sum(psd_mci_preview[i, 2:180]) # [delta/broad; alpha/broad]

patients_stages_mci = np.zeros((len(abeta_MCI_patients), 2)) # tau stage, abeta stage

patient_data = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\MCI_data\patients_common_indices_metrics.npz")
patient_psd_metrics = patient_data['psd_metrics'] #num_patients, metrics [Delta/Broad, Theta/Broad, Alpha/Broad, Beta/Broad]
patient_fc_metrics = patient_data['fc_metrics'] #num_patients, metrics [regular abs broad; regular abs delta; regular abs theta; regular abs alpha; regular abs beta]
stages_data = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\Staging_del_25_01_(new_staging)\stage_metrics_del_25_01_(new_staging).npz")
stages_psd_metrics = stages_data['psd_metrics'] #tau stage, abeta stage, metrics [Delta/Broad, Theta/Broad, Alpha/Broad, Beta/Broad]
stages_fc_metrics = stages_data['fc_metrics'] #tau stage, abeta stage, metrics [regular abs broad; regular abs delta; regular abs theta; regular abs alpha; regular abs beta]
print(stages_fc_metrics.shape)
distances = np.zeros((len(patient_fc_metrics[:,0]), len(stages_fc_metrics[:,0,0]), len(stages_fc_metrics[0,:,0])))
for p_idx in range(len(patient_fc_metrics[:,0])):
    for tau_idx in range(len(stages_fc_metrics[:,0,0])):
            for abeta_idx in range(len(stages_fc_metrics[0,:,0])):
                distances[p_idx, tau_idx, abeta_idx] = (np.abs(patient_psd_metrics[p_idx, 0] - stages_psd_metrics[tau_idx, abeta_idx, 0]) +
                                                        np.abs(patient_psd_metrics[p_idx, 2] - stages_psd_metrics[tau_idx, abeta_idx, 2]) +
                                                        np.abs(patient_fc_metrics[p_idx, 1] - stages_fc_metrics[tau_idx, abeta_idx, 1]) +
                                                        np.abs(patient_fc_metrics[p_idx, 2] - stages_fc_metrics[tau_idx, abeta_idx, 2]) +
                                                        np.abs(patient_fc_metrics[p_idx, 3] - stages_fc_metrics[tau_idx, abeta_idx, 3])
                                                       )

distances_abeta = np.zeros((len(patient_fc_metrics[:,0]), len(stages_fc_metrics[0,:,0])))
distances_tau = np.zeros((len(patient_fc_metrics[:,0]), len(stages_fc_metrics[:,0,0])))
for p_idx in range(len(patient_fc_metrics[:,0])):
    for tau_idx in range(len(stages_fc_metrics[:,0,0])):
            distances_tau[p_idx, tau_idx] = (np.abs(patient_psd_metrics[p_idx, 0] - stages_psd_metrics[tau_idx, 0, 0]) +
                                             np.abs(patient_fc_metrics[p_idx, 1] - stages_fc_metrics[tau_idx, 0, 1]) 
                                                   )
            
for p_idx in range(len(patient_fc_metrics[:,0])):
    for abeta_idx in range(len(stages_fc_metrics[0,:,0])):
        distances_abeta[p_idx, abeta_idx] = (np.abs(patient_psd_metrics[p_idx, 2] - stages_psd_metrics[0, abeta_idx, 2]) +
                                             np.abs(patient_fc_metrics[p_idx, 2] - stages_fc_metrics[0, abeta_idx, 2]) +
                                             np.abs(patient_fc_metrics[p_idx, 3] - stages_fc_metrics[0, abeta_idx, 3]) 
                                                   )

abeta_level = np.loadtxt(r"C:\Users\User\OneDrive - University of Pisa\Desktop\CSF_abeta.csv", skiprows=1)
tau_level = np.loadtxt(r"C:\Users\User\OneDrive - University of Pisa\Desktop\CSF_p_tau.csv", skiprows=1)
prot_lev = np.column_stack((abeta_level, tau_level))
patients_stages = np.zeros((len(prot_lev), 2)) # tau stage, abeta stage
######## ora determino lo stage dei pazienti in funzione dei protein levels determinati a partire dall'articolo"Clinical and biomarker trajectories in sporadic Alzheimer’s
######## disease: A longitudinal study" di Alzheimer's and Dementia
#CSF abeta level at stage I:  0.2318534187814261
#CSF abeta level at stage II:  0.10060892803728635
#CSF abeta level at stage III:  0.05000000000000001
#CSF tau level at stage I:  0.5553246833941619
#CSF tau level at stage II:  0.8543703680827017
#CSF tau level at stage III:  0.9500000000000001
######## per fare le proporzioni con i nostri livelli di proteine abbiamom soltanto il livello soglia, che supponiamo sia quello di cutoff
######## p-tau >= 52 mg/L che corrisponde a 30.29 (inferred)
######## abeta <= 813 mg/L che corrisponde a 192 (labeled in the article); la percentuale di decrescita della proteina alla quale è fissato il cutoff è 0.20618514077810396
######## adesso dobbiamo risalire ai valori corrispondenti di abeta e tau nei 3 stages
######## partiamo con i valori massimo di abeta e minimo di tau 
######## il valore massimo di abeta sarà dato dalla proporzione x:813 = 241.87:192 --> x = 1024.17
######## di conseguenza possiamo calcolare il valore minimo di abeta x:1024.17 = 130.53:241.87 --> x = 552.71
######## allo stesso modo il valore minimo di tau sarà dato da x:52 = 47.07:30.29 --> x = 80.81
######## quindi i delta del nostro modello saranno: abeta 1024.17 - 552.71 = 471.46
########                                            tau 80.81 - 52 = 28.81

abeta_delta = 1024.17 - 552.71
abeta_stage_1 = 552.71 + abeta_delta*0.2318534187814261
abeta_stage_2 = 552.71 + abeta_delta*0.10060892803728635
abeta_stage_3 = 552.71 + abeta_delta*0.05000000000000001

tau_delta = 80.81 - 28.81
tau_stage_1 = 28.81 + tau_delta*0.5553246833941619
tau_stage_2 = 28.81 + tau_delta*0.8543703680827017
tau_stage_3 = 28.81 + tau_delta*0.9500000000000001

#########con questa divisione in stage la performance dell'algoritmo sale a 10 match e 4 mismatch di 1 (su 30 confronti totali)
for i in range(len(patient_fc_metrics[:,0])):
    if (prot_lev[i,0] == 0) & (prot_lev[i,1] == 0):
        patients_stages[i,0] = np.nan
        patients_stages[i,1] = np.nan
    #prot_lev[i,0] < abeta_stage_1
    if (prot_lev[i,0] > abeta_stage_1) & (prot_lev[i,1] < tau_stage_1):
        patients_stages[i,0] = 0
        patients_stages[i,1] = 0

    if (prot_lev[i,0] > abeta_stage_1) & (tau_stage_2 > prot_lev[i,1] >= tau_stage_1):
        patients_stages[i,0] = 3
        patients_stages[i,1] = 3

    if (prot_lev[i,0] > abeta_stage_1) & (tau_stage_3 > prot_lev[i,1] >= tau_stage_2):
        patients_stages[i,0] = 4
        patients_stages[i,1] = 3
       
    if (prot_lev[i,0] > abeta_stage_1) & (prot_lev[i,1] >= tau_stage_3):
        patients_stages[i,0] = 5
        patients_stages[i,1] = 3

    #abeta_stage_2 > prot_lev[i,0] >= abeta_stage_1
    if (abeta_stage_2 < prot_lev[i,0] <= abeta_stage_1) & (prot_lev[i,1] < tau_stage_1):
        patients_stages[i,0] = 2
        patients_stages[i,1] = 0

    if (abeta_stage_2 < prot_lev[i,0] <= abeta_stage_1) & (tau_stage_2 > prot_lev[i,1] >= tau_stage_1):
        patients_stages[i,0] = 3
        patients_stages[i,1] = 1

    if (abeta_stage_2 < prot_lev[i,0] <= abeta_stage_1) & (tau_stage_3 > prot_lev[i,1] >= tau_stage_2):
        patients_stages[i,0] = 4
        patients_stages[i,1] = 2
        
    if (abeta_stage_2 < prot_lev[i,0] <= abeta_stage_1) & (prot_lev[i,1] >= tau_stage_3):
        patients_stages[i,0] = 5
        patients_stages[i,1] = 3

    #abeta_stage_3 > prot_lev[i,0] >= abeta_stage_2
    if (abeta_stage_3 < prot_lev[i,0] <= abeta_stage_2) & (prot_lev[i,1] < tau_stage_1):
        patients_stages[i,0] = 2
        patients_stages[i,1] = 0

    if (abeta_stage_3 < prot_lev[i,0] <= abeta_stage_2) & (tau_stage_2 > prot_lev[i,1] >= tau_stage_1):
        patients_stages[i,0] = 3
        patients_stages[i,1] = 1

    if (abeta_stage_3 < prot_lev[i,0] <= abeta_stage_2) & (tau_stage_3 > prot_lev[i,1] >= tau_stage_2):
        patients_stages[i,0] = 4
        patients_stages[i,1] = 2
        
    if (abeta_stage_3 < prot_lev[i,0] <= abeta_stage_2) & (prot_lev[i,1] >= tau_stage_3):
        patients_stages[i,0] = 5
        patients_stages[i,1] = 3

    #prot_lev[i,0] >= abeta_stage_3
    if (prot_lev[i,0] <= abeta_stage_3) & (prot_lev[i,1] < tau_stage_1):
        patients_stages[i,0] = 3
        patients_stages[i,1] = 0

    if (prot_lev[i,0] <= abeta_stage_3) & (tau_stage_2 > prot_lev[i,1] >= tau_stage_1):
        patients_stages[i,0] = 3
        patients_stages[i,1] = 1

    if (prot_lev[i,0] <= abeta_stage_3) & (tau_stage_3 > prot_lev[i,1] >= tau_stage_2):
        patients_stages[i,0] = 3
        patients_stages[i,1] = 2
        
    if (prot_lev[i,0] <= abeta_stage_3) & (prot_lev[i,1] >= tau_stage_3):
        patients_stages[i,0] = 3
        patients_stages[i,1] = 3

    idx_abeta = np.argmin(distances_abeta[i, :])
    idx_tau = np.argmin(distances_tau[i,:])
    print(i, ': ', prot_lev[i], '; ', patients_stages[i], '; ', idx_abeta, ' ', idx_tau)

    idx_flat = np.argmin(distances[i, :, :])

    # 2. Convertilo in coordinate (riga, colonna)
    riga, colonna = np.unravel_index(idx_flat, distances[i, :, :].shape)
    print(i, ': ', prot_lev[i], '; ', patients_stages[i], '; ', riga, ' ', colonna, '\n')

tau_stage_1 = 20.00 #28.81 + tau_delta*0.5553246833941619
tau_stage_2 = 28.81 #+ tau_delta*0.8543703680827017
tau_stage_3 = 28.81 + tau_delta*0.5553246833941619
tau_stage_4 = 28.81 + tau_delta*0.8543703680827017
tau_stage_5 = 28.81 + tau_delta*0.9500000000000001
print ('INIZIO NEW STAGING')
for i in range(len(patient_fc_metrics[:,0])):
    if (prot_lev[i,0] == 0) & (prot_lev[i,1] == 0):
        patients_stages[i,0] = np.nan
        patients_stages[i,1] = np.nan

    if (prot_lev[i,1] < tau_stage_1):
        patients_stages[i,1] = 0
    if (tau_stage_2 > prot_lev[i,1] >= tau_stage_1):
        patients_stages[i,1] = 1
    if (tau_stage_3 > prot_lev[i,1] >= tau_stage_2):
        patients_stages[i,1] = 2
    if (tau_stage_4 > prot_lev[i,1] >= tau_stage_3):
        patients_stages[i,1] = 3
    if (tau_stage_5 > prot_lev[i,1] >= tau_stage_4):
        patients_stages[i,1] = 4
    if (prot_lev[i,1] >= tau_stage_5):
        patients_stages[i,1] = 5

    if (prot_lev[i,0] > abeta_stage_1) & (prot_lev[i,1] < tau_stage_2):
        patients_stages[i,0] = 0

    if (prot_lev[i,0] > abeta_stage_1) & (prot_lev[i,1] >= tau_stage_2):
        patients_stages[i,0] = 5

    if (abeta_stage_2 < prot_lev[i,0] <= abeta_stage_1) & (prot_lev[i,1] < tau_stage_2):
        patients_stages[i,0] = 1

    if (abeta_stage_2 < prot_lev[i,0] <= abeta_stage_1) & (prot_lev[i,1] >= tau_stage_2):
        patients_stages[i,0] = 5

    if (abeta_stage_3 < prot_lev[i,0] <= abeta_stage_2) & (prot_lev[i,1] < tau_stage_2):
        patients_stages[i,0] = 2

    if (abeta_stage_3 < prot_lev[i,0] <= abeta_stage_2) & (prot_lev[i,1] >= tau_stage_2):
        patients_stages[i,0] = 4

    if (prot_lev[i,0] <= abeta_stage_3):
        patients_stages[i,0] = 3

    idx_abeta = np.argmin(distances_abeta[i, :])
    idx_tau = np.argmin(distances_tau[i,:])
    if prot_lev[i,0] != 0:
        print(i, ': ', prot_lev[i], '; ', patients_stages[i], '; ', idx_abeta, ' ', idx_tau)

    idx_flat = np.argmin(distances[i, :, :])

    # 2. Convertilo in coordinate (riga, colonna)
    riga, colonna = np.unravel_index(idx_flat, distances[i, :, :].shape)
    if prot_lev[i,0] != 0:
        print(i, ': ', prot_lev[i], '; ', patients_stages[i], '; ', riga, ' ', colonna, '\n')
print ('FINE NEW STAGING')
########## ovviamente questi sono 3 stage mentre noi ne abbiamo 5, per come abbiamo strutturato i nostri stages
########## la corrispondenda stages_reali vs stages_simulati sarà la seguente:
########## stage_reale 1 --> stage_simulato 1
########## stage_reale 2 --> stage_simulato 2
########## stage_reale 3 --> stage_simulato 3,4,5

print(stages_fc_metrics.shape)
distances_mci = np.zeros((len(abeta_MCI_patients), 6, 6))
for p_idx in range(len(abeta_MCI_patients)):
    for tau_idx in range(6):
        for abeta_idx in range(6):
            distances_mci[p_idx, tau_idx, abeta_idx] = (np.abs(mci_preview_metrics[p_idx,0] - stages_fc_metrics[tau_idx, abeta_idx, 2]) + 
                                                   np.abs(mci_preview_metrics[p_idx, 1] - stages_psd_metrics[tau_idx, abeta_idx, 0]) +
                                                   np.abs(mci_preview_metrics[p_idx, 2] - stages_psd_metrics[tau_idx, abeta_idx, 1]) 
                                                   )
            
for i in range(len(abeta_MCI_patients)):
    if (prot_lev_mci[i,0] != np.nan) & (prot_lev_mci[i,1] != np.nan):
        match prot_lev_mci[i,0]:
            case x if x >= abeta_stage_1:
                patients_stages_mci[i,0] = 0
            case x if abeta_stage_1 > x >= abeta_stage_2:
                patients_stages_mci[i,0] = 1
            case x if abeta_stage_2 > x >= abeta_stage_3:
                patients_stages_mci[i,0] = 2
            case _:
                patients_stages_mci[i,0] = 3
        match prot_lev_mci[i,1]:
            case x if x <= tau_stage_1:
                patients_stages_mci[i,1] = 0
            case x if tau_stage_2 > x >= tau_stage_1:
                patients_stages_mci[i,1] = 1
            case x if tau_stage_3 > x >= tau_stage_2:
                patients_stages_mci[i,1] = 2
            case _:
                patients_stages_mci[i,1] = 3
    else:
        patients_stages_mci[i,0] = np.nan
        patients_stages_mci[i,1] = np.nan

    idx_flat = np.argmin(distances_mci[i, :, :])

    # 2. Convertilo in coordinate (riga, colonna)
    riga, colonna = np.unravel_index(idx_flat, distances_mci[i, :, :].shape)
    if (prot_lev_mci[i,0] < 2000):
        print(i, ': ', prot_lev_mci[i], '; ', patients_stages_mci[i], '; ', riga, ' ', colonna)
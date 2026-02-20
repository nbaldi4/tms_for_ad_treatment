import pandas as pd
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import global_variable_creation as gv
from sklearn.feature_selection import mutual_info_regression

from data_analysis_utils import real_vs_stages_PSD, real_vs_stages_FC, to_common_indices, macro_fc, analyze_regular_data, analyze_reduced_data


# Rimuove il limite di larghezza della colonna (None = nessun limite)
pd.set_option('display.max_colwidth', None)

patient_data = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\MCI_data\patients_common_indices_metrics.npz")
patient_psd_metrics = patient_data['psd_metrics'] #num_patients, metrics [delta/broad; alpha/broad]
patient_fc_metrics = patient_data['fc_metrics'] #num_patients, metrics [regular abs delta; regular abs theta; regular abs alpha]
patient_metrics = np.zeros((len(patient_psd_metrics[:,0]), len(patient_psd_metrics[0,:]) + len(patient_fc_metrics[0,:])))
for i in range(len(patient_psd_metrics[:,0])):
    patient_metrics[i,0] = patient_psd_metrics[i,0] #delta/broad
    patient_metrics[i,1] = patient_psd_metrics[i,1] #theta/broad
    patient_metrics[i,2] = patient_psd_metrics[i,2] #alpha/broad
    patient_metrics[i,3] = patient_psd_metrics[i,3] #beta/broad
    patient_metrics[i,4] = patient_fc_metrics[i,0] #regular abs broad
    patient_metrics[i,5] = patient_fc_metrics[i,1] #regular abs delta
    patient_metrics[i,6] = patient_fc_metrics[i,2] #regular abs theta
    patient_metrics[i,7] = patient_fc_metrics[i,3] #regular abs alpha
    patient_metrics[i,8] = patient_fc_metrics[i,4] #regular abs beta

abeta_level = np.loadtxt(r"C:\Users\User\OneDrive - University of Pisa\Desktop\CSF_abeta.csv", skiprows=1)
tau_level = np.loadtxt(r"C:\Users\User\OneDrive - University of Pisa\Desktop\CSF_p_tau.csv", skiprows=1)
prot_lev = np.column_stack((abeta_level, tau_level))

def calcola_ranking_mi(df_metriche, target, nome_target="Proteina"):

    mi_scores = mutual_info_regression(
        df_metriche.values,   # X: (n_samples, n_features)
        target,               # y: (n_samples,)
        discrete_features=False,
        random_state=42
    )

    mi_results = pd.DataFrame({
        'Metrica': df_metriche.columns,
        'MI_Score': mi_scores
    }).sort_values(by='MI_Score', ascending=False)

    print(f"\n--- Ranking Mutual Information per {nome_target} ---")
    print(mi_results.to_string(index=False))

    return mi_results

# Analisi per Abeta
# Metrics name 
nomi_delle_mie_metriche = [    'delta/broad', 'theta/broad', 'alpha/broad', 'beta/broad', 'regular abs broad', 'regular abs delta', 
                           'regular abs theta', 'regular abs alpha', 'regular abs beta'
                          ]

#DataFrame convertion
patient_metrics_df = pd.DataFrame(patient_metrics, columns=nomi_delle_mie_metriche)
mi_abeta = calcola_ranking_mi(patient_metrics_df, abeta_level[:len(patient_psd_metrics[:,0])], "Abeta CSF")
print(mi_abeta) 
#le metriche per abeta sono: 
#regular abs theta  0.046920
#regular abs beta  0.028926
#regular abs broad  0.022283
#alpha/broad  0.009598
#regular abs alpha  0.005384
#delta/broad  0.000000
#theta/broad  0.000000
#beta/broad  0.000000
#regular abs delta  0.000000

mi_tau = calcola_ranking_mi(patient_metrics_df, tau_level[:len(patient_psd_metrics[:,0])], "Tau CSF")
print(mi_tau)
#le metriche con la mi più alta per tau sono:
#alpha/broad  0.047697
#regular abs broad  0.045865
#regular abs alpha  0.037022
#regular abs beta  0.014023
#theta/broad  0.013335
#regular abs theta  0.010902
#delta/broad  0.000000
#beta/broad  0.000000
#regular abs delta  0.000000

# Analisi per Tau
#mi_tau = calcola_ranking_mi(df_metriche, df_csf['Tau_CSF'], "Tau CSF")

stages_data = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\Staging_del_25_01_(new_staging)\stage_metrics_del_25_01_(new_staging).npz")
stages_psd_metrics = stages_data['psd_metrics'] #tau stage, abeta stage, metrics ['delta/broad', 'theta/broad', 'alpha/broad', 'beta/broad']
stages_fc_metrics = stages_data['fc_metrics'] #tau stage, abeta stage, metrics ['regular abs broad', 'regular abs delta', 'regular abs theta', 'regular abs alpha', 'regular abs beta']
print(stages_fc_metrics.shape)
distances = np.zeros((len(patient_fc_metrics[:,0]), len(stages_fc_metrics[:,0,0]), len(stages_fc_metrics[0,:,0])))
for p_idx in range(len(patient_fc_metrics[:,0])):
    for tau_idx in range(len(stages_fc_metrics[:,0,0])):
        for abeta_idx in range(len(stages_fc_metrics[0,:,0])):
            distances[p_idx, tau_idx, abeta_idx] = (np.abs(patient_psd_metrics[p_idx, 2] - stages_psd_metrics[tau_idx, abeta_idx, 2]) + 
                                                    np.abs(patient_fc_metrics[p_idx, 0] - stages_fc_metrics[tau_idx, abeta_idx, 0]) +
                                                    np.abs(patient_fc_metrics[p_idx, 2] - stages_fc_metrics[tau_idx, abeta_idx, 2]) +
                                                    np.abs(patient_fc_metrics[p_idx, 3] - stages_fc_metrics[tau_idx, abeta_idx, 3]) +
                                                    np.abs(patient_fc_metrics[p_idx, 4] - stages_fc_metrics[tau_idx, abeta_idx, 4]) 
                                                )
            
distances_tau = np.zeros((len(patient_fc_metrics[:,0]), len(stages_fc_metrics[:,0,0])))
for p_idx in range(len(patient_fc_metrics[:,0])):
    for tau_idx in range(len(stages_fc_metrics[:,0,0])):
        distances_tau[p_idx, tau_idx] = (np.abs(patient_psd_metrics[p_idx, 2] - stages_psd_metrics[tau_idx, abeta_idx, 2]) + 
                                                   np.abs(patient_fc_metrics[p_idx, 0] - stages_fc_metrics[tau_idx, abeta_idx, 0]) +
                                                   np.abs(patient_fc_metrics[p_idx, 3] - stages_fc_metrics[tau_idx, abeta_idx, 3])
                                                   )
            
distances_abeta = np.zeros((len(patient_fc_metrics[:,0]), len(stages_fc_metrics[0,:,0])))
for p_idx in range(len(patient_fc_metrics[:,0])):
    for abeta_idx in range(len(stages_fc_metrics[0,:,0])):
        distances_abeta[p_idx, abeta_idx] = (np.abs(patient_fc_metrics[p_idx, 0] - stages_fc_metrics[tau_idx, abeta_idx, 0]) +
                                                   np.abs(patient_fc_metrics[p_idx, 2] - stages_fc_metrics[tau_idx, abeta_idx, 2])+
                                                   np.abs(patient_fc_metrics[p_idx, 4] - stages_fc_metrics[tau_idx, abeta_idx, 4])
        )
            

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
tau_error = 0
abeta_error = 0
correclty_id = 0
one_error = 0
two_error = 0
three_error = 0
four_error = 0
five_error = 0
for i in range(len(patient_fc_metrics[:,0])):
    if (prot_lev[i,0] == 0) & (prot_lev[i,1] == 0):
        patients_stages[i,0] = np.nan
        patients_stages[i,1] = np.nan
    #prot_lev[i,0] < abeta_stage_1
    if (prot_lev[i,0] < abeta_stage_1) & (prot_lev[i,1] < tau_stage_1):
        patients_stages[i,0] = 0
        patients_stages[i,1] = 0

    if (prot_lev[i,0] < abeta_stage_1) & (tau_stage_2 > prot_lev[i,1] >= tau_stage_1):
        patients_stages[i,0] = 3
        patients_stages[i,1] = 3

    if (prot_lev[i,0] < abeta_stage_1) & (tau_stage_3 > prot_lev[i,1] >= tau_stage_2):
        patients_stages[i,0] = 4
        patients_stages[i,1] = 3
       
    if (prot_lev[i,0] < abeta_stage_1) & (prot_lev[i,1] >= tau_stage_3):
        patients_stages[i,0] = 5
        patients_stages[i,1] = 3

    #abeta_stage_2 > prot_lev[i,0] >= abeta_stage_1
    if (abeta_stage_2 > prot_lev[i,0] >= abeta_stage_1) & (prot_lev[i,1] < tau_stage_1):
        patients_stages[i,0] = 2
        patients_stages[i,1] = 0

    if (abeta_stage_2 > prot_lev[i,0] >= abeta_stage_1) & (tau_stage_2 > prot_lev[i,1] >= tau_stage_1):
        patients_stages[i,0] = 3
        patients_stages[i,1] = 1

    if (abeta_stage_2 > prot_lev[i,0] >= abeta_stage_1) & (tau_stage_3 > prot_lev[i,1] >= tau_stage_2):
        patients_stages[i,0] = 4
        patients_stages[i,1] = 2
        
    if (abeta_stage_2 > prot_lev[i,0] >= abeta_stage_1) & (prot_lev[i,1] >= tau_stage_3):
        patients_stages[i,0] = 5
        patients_stages[i,1] = 3

    #abeta_stage_3 > prot_lev[i,0] >= abeta_stage_2
    if (abeta_stage_3 > prot_lev[i,0] >= abeta_stage_2) & (prot_lev[i,1] < tau_stage_1):
        patients_stages[i,0] = 2
        patients_stages[i,1] = 0

    if (abeta_stage_3 > prot_lev[i,0] >= abeta_stage_2) & (tau_stage_2 > prot_lev[i,1] >= tau_stage_1):
        patients_stages[i,0] = 3
        patients_stages[i,1] = 1

    if (abeta_stage_3 > prot_lev[i,0] >= abeta_stage_2) & (tau_stage_3 > prot_lev[i,1] >= tau_stage_2):
        patients_stages[i,0] = 4
        patients_stages[i,1] = 2
        
    if (abeta_stage_3 > prot_lev[i,0] >= abeta_stage_2) & (prot_lev[i,1] >= tau_stage_3):
        patients_stages[i,0] = 5
        patients_stages[i,1] = 3

    #prot_lev[i,0] >= abeta_stage_3
    if (prot_lev[i,0] >= abeta_stage_3) & (prot_lev[i,1] < tau_stage_1):
        patients_stages[i,0] = 3
        patients_stages[i,1] = 0

    if (prot_lev[i,0] >= abeta_stage_3) & (tau_stage_2 > prot_lev[i,1] >= tau_stage_1):
        patients_stages[i,0] = 3
        patients_stages[i,1] = 1

    if (prot_lev[i,0] >= abeta_stage_3) & (tau_stage_3 > prot_lev[i,1] >= tau_stage_2):
        patients_stages[i,0] = 3
        patients_stages[i,1] = 2
        
    if (prot_lev[i,0] >= abeta_stage_3) & (prot_lev[i,1] >= tau_stage_3):
        patients_stages[i,0] = 3
        patients_stages[i,1] = 3

    idx_flat = np.argmin(distances[i, :, :])

    # 2. Convertilo in coordinate (riga, colonna)
    riga, colonna = np.unravel_index(idx_flat, distances[i, :, :].shape)
    #riga = np.argmin(distances_abeta[i, :])
    #colonna = np.argmin(distances_tau[i, :])
    if prot_lev[i,0] != 0:
        #print(i, ': ', prot_lev[i], '; ', patients_stages[i], '; ', riga, ' ', colonna, '; ', patients_stages[i,0]-riga, ' ', patients_stages[i,1]-colonna)
        print("{:3d} : {:>5} {:>5} ; {:>10} ; {:>7}".format(
            i,
            str(prot_lev[i, 0]),
            str(prot_lev[i, 1]),
            str(patients_stages[i]),
            f"{riga} {colonna}"
        ))
        abeta_error += patients_stages[i,0]-riga
        tau_error += patients_stages[i,1]-colonna
        if np.abs(patients_stages[i,1]-colonna) == 0:
            correclty_id += 1
        if np.abs(patients_stages[i,1]-colonna) == 1:
            one_error += 1
        if np.abs(patients_stages[i,1]-colonna) == 2:
            two_error += 1
        if np.abs(patients_stages[i,1]-colonna) == 3:
            three_error += 1
        if np.abs(patients_stages[i,1]-colonna) == 4:
            four_error += 1
        if np.abs(patients_stages[i,1]-colonna) == 5:
            five_error += 1 

print('corretly identified patient: ', correclty_id)
print('1 stage mismatch: ', one_error)
print('2 stage mismatch: ', two_error)
print('3 stage mismatch: ', three_error)
print('4 stage mismatch: ', four_error)
print('5 stage mismatch: ', five_error)
print(abeta_error)
print(tau_error)
########## ovviamente questi sono 3 stage mentre noi ne abbiamo 5, per come abbiamo strutturato i nostri stages
########## la corrispondenda stages_reali vs stages_simulati sarà la seguente:
########## stage_reale 1 --> stage_simulato 1
########## stage_reale 2 --> stage_simulato 2
########## stage_reale 3 --> stage_simulato 3,4,5
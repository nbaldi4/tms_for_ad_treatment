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

mci_data = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Patients_highlights\mci_summary_stats_new.npz", allow_pickle=True)
mci_fc_abs_mean = mci_data['mci_fc_abs_mean']
mci_psd_rel_pow = mci_data['mci_psd_rel_pow']
mci_csf_values = mci_data['mci_csf']

stages_fc_data = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Staging_highlights\Stages_increase_b_simulated_stages_fc.npz", allow_pickle=True)
fc_matrices=stages_fc_data['fc_matrices']
stages_psd_data = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Staging_highlights\Stages_increase_b_simulated_stages_psd.npz", allow_pickle=True)
psd_arrays=stages_psd_data['psd_arrays']
psd_freqs=stages_psd_data['psd_freqs']

abeta_stages = 6
tau_stages = 6
patients_number = 59

stages_metrics = np.zeros((6, 6, 9))
for abeta in range(abeta_stages):
    for tau in range(tau_stages):
        stages_metrics[abeta,tau,0] = np.mean(np.abs(fc_matrices[abeta,tau,0,0]))
        stages_metrics[abeta,tau,1] = np.mean(np.abs(fc_matrices[abeta,tau,1,0]))
        stages_metrics[abeta,tau,2] = np.mean(np.abs(fc_matrices[abeta,tau,2,0]))
        stages_metrics[abeta,tau,3] = np.mean(np.abs(fc_matrices[abeta,tau,3,0]))
        stages_metrics[abeta,tau,4] = np.mean(np.abs(fc_matrices[abeta,tau,4,0]))
        freqs = psd_freqs[abeta,tau,0]
        psd_broad = np.sum(psd_arrays[abeta,tau,0])                               #broad
        psd_delta = np.sum(psd_arrays[abeta,tau,0][(freqs>= 0.5) & (freqs < 4)])  #delta
        psd_theta = np.sum(psd_arrays[abeta,tau,0][(freqs >= 4) & (freqs < 8)])   #theta
        psd_alpha = np.sum(psd_arrays[abeta,tau,0][(freqs >= 8) & (freqs < 13)])  #alpha
        psd_beta = np.sum(psd_arrays[abeta,tau,0][(freqs >= 13) & (freqs < 30)])  #beta
        stages_metrics[abeta,tau,5] = psd_delta/psd_broad
        stages_metrics[abeta,tau,6] = psd_theta/psd_broad
        stages_metrics[abeta,tau,7] = psd_alpha/psd_broad
        stages_metrics[abeta,tau,8] = psd_beta/psd_broad

stages_data = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\Staging_del_25_01_(new_staging)\stage_metrics_del_25_01_(new_staging).npz")
stages_psd_metrics = stages_data['psd_metrics'] #tau stage, abeta stage, metrics ['delta/broad', 'theta/broad', 'alpha/broad', 'beta/broad']
stages_fc_metrics = stages_data['fc_metrics'] #tau stage, abeta stage, metrics ['regular abs broad', 'regular abs delta', 'regular abs theta', 'regular abs alpha', 'regular abs beta']
#print(stages_fc_metrics.shape)
#print(stages_psd_metrics.shape)cs
#print(stages_metrics[2,3,:])
#attentione che nello staging del 25_01 l'ordine è tau_abeta e non abeta_tau
#for abeta in range(abeta_stages):
#    for tau in range(tau_stages):
#        stages_metrics[abeta,tau,:5] = stages_fc_metrics[tau,abeta,:]
#        stages_metrics[abeta,tau,5:] = stages_psd_metrics[tau,abeta,:]
#print(stages_metrics[2,3,:])

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
fc_metrics_name = ['regular abs broad', 'regular abs delta', 'regular abs theta', 'regular abs alpha', 'regular abs beta']
psd_metrics_name = ['delta/broad', 'theta/broad', 'alpha/broad', 'beta/broad']

#DataFrame convertion
patient_fc_metrics_df = pd.DataFrame(mci_fc_abs_mean, columns=fc_metrics_name)
mi_abeta_fc = calcola_ranking_mi(patient_fc_metrics_df, mci_csf_values[:,0], "Abeta CSF")
print(mi_abeta_fc)
patient_psd_metrics_df = pd.DataFrame(mci_psd_rel_pow, columns=psd_metrics_name)
mi_abeta_psd = calcola_ranking_mi(patient_psd_metrics_df, mci_csf_values[:,0], "Abeta CSF")
print(mi_abeta_psd)
"""
--- Ranking Mutual Information per Abeta CSF ---
          Metrica  MI_Score
regular abs broad  0.122744
regular abs delta  0.109253
regular abs theta  0.035403
regular abs alpha  0.000000
 regular abs beta  0.000000
theta/broad  0.130499
 beta/broad  0.012448
delta/broad  0.000000
alpha/broad  0.000000
"""
mi_tau_fc = calcola_ranking_mi(patient_fc_metrics_df, mci_csf_values[:,1], "Tau CSF")
print(mi_tau_fc)
mi_tau_psd = calcola_ranking_mi(patient_psd_metrics_df, mci_csf_values[:,1], "Tau CSF")
print(mi_tau_psd)
"""
--- Ranking Mutual Information per Tau CSF ---
          Metrica  MI_Score
 regular abs beta  0.046003
regular abs delta  0.033562
regular abs theta  0.018514
regular abs broad  0.000000
regular abs alpha  0.000000
       Metrica  MI_Score
 beta/broad  0.025874
delta/broad  0.000000
theta/broad  0.000000
alpha/broad  0.000000
"""
distances = np.zeros((patients_number, abeta_stages, tau_stages))
for p_idx in range(patients_number):
    for abeta_idx in range(abeta_stages):
        for tau_idx in range(tau_stages):
            distances[p_idx, abeta_idx, tau_idx] = (np.abs(mci_psd_rel_pow[p_idx, 1] - stages_metrics[abeta_idx, tau_idx, 1]) + # theta/broad
                                                    np.abs(mci_psd_rel_pow[p_idx, 3] - stages_metrics[abeta_idx, tau_idx, 3]) + # beta/broad
                                                    np.abs(mci_fc_abs_mean[p_idx, 0] - stages_metrics[abeta_idx, tau_idx, 0]) + # regular abs broad
                                                    np.abs(mci_fc_abs_mean[p_idx, 1] - stages_metrics[abeta_idx, tau_idx, 1]) + # regular abs delta
                                                    np.abs(mci_fc_abs_mean[p_idx, 4] - stages_metrics[abeta_idx, tau_idx, 4])   # regular abs beta
                                                )
            
distances_tau = np.zeros((patients_number, abeta_stages, tau_stages))
for p_idx in range(patients_number):
    for abeta_idx in range(abeta_stages):
        for tau_idx in range(tau_stages):
            distances_tau[p_idx, abeta_idx, tau_idx] = (np.abs(mci_psd_rel_pow[p_idx, 3] - stages_metrics[abeta_idx, tau_idx, 3]) + # beta/broad
                                                        np.abs(mci_fc_abs_mean[p_idx, 1] - stages_metrics[abeta_idx, tau_idx, 1]) + # regular abs delta
                                                        np.abs(mci_fc_abs_mean[p_idx, 4] - stages_metrics[abeta_idx, tau_idx, 4])   # regular abs beta
                                                    )
            
distances_abeta = np.zeros((patients_number, abeta_stages, tau_stages))
for p_idx in range(patients_number):
    for abeta_idx in range(abeta_stages):
        for tau_idx in range(tau_stages):
            distances_abeta[p_idx, abeta_idx, tau_idx] = (np.abs(mci_psd_rel_pow[p_idx, 1] - stages_metrics[abeta_idx, tau_idx, 1]) + # theta/broad
                                                          np.abs(mci_fc_abs_mean[p_idx, 0] - stages_metrics[abeta_idx, tau_idx, 0]) + # regular abs broad
                                                          np.abs(mci_fc_abs_mean[p_idx, 1] - stages_metrics[abeta_idx, tau_idx, 1])   # regular abs delta
                                                        )
            

patients_stages = np.zeros((patients_number, 2)) # tau stage, abeta stage
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

tau_delta = 80.81 - 52
tau_stage_1 = 52 + tau_delta*0.5553246833941619
tau_stage_2 = 52 + tau_delta*0.8543703680827017
tau_stage_3 = 52 + tau_delta*0.9500000000000001


abeta_delta = np.max(mci_csf_values[:,0]) - 813
abeta_stage_1 = 552.71 + abeta_delta*0.2318534187814261
abeta_stage_2 = 552.71 + abeta_delta*0.10060892803728635
abeta_stage_3 = 552.71 + abeta_delta*0.05000000000000001
#print(abeta_stage_1)
#print(abeta_stage_2)
#print(abeta_stage_3)

tau_delta = np.max(mci_csf_values[:,1]) - 52 #np.min(mci_csf_values[:,1])
tau_stage_1 = 52 + tau_delta*0.5553246833941619
tau_stage_2 = 52 + tau_delta*0.8543703680827017
tau_stage_3 = 52 + tau_delta*0.9500000000000001
#print(tau_stage_1)
#print(tau_stage_2)
#print(tau_stage_3)

#########con questa divisione in stage la performance dell'algoritmo sale a 10 match e 4 mismatch di 1 (su 30 confronti totali)
sep_error = 0
contemp_error = 0
sep_correclty_id = 0
cont_correclty_id = 0
cont_correclty_id_abeta = 0
cont_correclty_id_tau = 0
sep_correclty_id_abeta = 0
sep_correclty_id_tau = 0
one_error = 0
two_error = 0
three_error = 0
four_error = 0
five_error = 0
for i in range(len(mci_csf_values[:,0])):
    # abeta < abeta_stage_1
    if (mci_csf_values[i,0] > abeta_stage_1) & (mci_csf_values[i,1] < tau_stage_1):
        patients_stages[i,0] = 0
        patients_stages[i,1] = 0

    if (mci_csf_values[i,0] > abeta_stage_1) & (tau_stage_2 > mci_csf_values[i,1] >= tau_stage_1):
        patients_stages[i,0] = 5
        patients_stages[i,1] = 1

    if (mci_csf_values[i,0] > abeta_stage_1) & (tau_stage_3 > mci_csf_values[i,1] >= tau_stage_2):
        patients_stages[i,0] = 5
        patients_stages[i,1] = 2
       
    if (mci_csf_values[i,0] > abeta_stage_1) & (mci_csf_values[i,1] >= tau_stage_3):
        patients_stages[i,0] = 5
        patients_stages[i,1] = 3

    #abeta_stage_2 < abeta <= abeta_stage_1
    if (abeta_stage_2 < mci_csf_values[i,0] <= abeta_stage_1) & (mci_csf_values[i,1] < tau_stage_1):
        patients_stages[i,0] = 1
        patients_stages[i,1] = 0

    if (abeta_stage_2 < mci_csf_values[i,0] <= abeta_stage_1) & (tau_stage_2 > mci_csf_values[i,1] >= tau_stage_1):
        patients_stages[i,0] = 4
        patients_stages[i,1] = 1

    if (abeta_stage_2 < mci_csf_values[i,0] <= abeta_stage_1) & (tau_stage_3 > mci_csf_values[i,1] >= tau_stage_2):
        patients_stages[i,0] = 4
        patients_stages[i,1] = 2
        
    if (abeta_stage_2 < mci_csf_values[i,0] <= abeta_stage_1) & (mci_csf_values[i,1] >= tau_stage_3):
        patients_stages[i,0] = 4
        patients_stages[i,1] = 3

    #abeta_stage_3 < abeta <= abeta_stage_2
    if (abeta_stage_3 < mci_csf_values[i,0] <= abeta_stage_2) & (mci_csf_values[i,1] < tau_stage_1):
        patients_stages[i,0] = 2
        patients_stages[i,1] = 0

    if (abeta_stage_3 < mci_csf_values[i,0] <= abeta_stage_2) & (tau_stage_2 > mci_csf_values[i,1] >= tau_stage_1):
        patients_stages[i,0] = 2
        patients_stages[i,1] = 1

    if (abeta_stage_3 < mci_csf_values[i,0] <= abeta_stage_2) & (tau_stage_3 > mci_csf_values[i,1] >= tau_stage_2):
        patients_stages[i,0] = 4
        patients_stages[i,1] = 2
        
    if (abeta_stage_3 < mci_csf_values[i,0] <= abeta_stage_2) & (mci_csf_values[i,1] >= tau_stage_3):
        patients_stages[i,0] = 4
        patients_stages[i,1] = 3

    #prot_lev[i,0] <= abeta_stage_3
    if (mci_csf_values[i,0] <= abeta_stage_3) & (mci_csf_values[i,1] < tau_stage_1):
        patients_stages[i,0] = 3
        patients_stages[i,1] = 0

    if (mci_csf_values[i,0] <= abeta_stage_3) & (tau_stage_2 > mci_csf_values[i,1] >= tau_stage_1):
        patients_stages[i,0] = 3
        patients_stages[i,1] = 1

    if (mci_csf_values[i,0] <= abeta_stage_3) & (tau_stage_3 > mci_csf_values[i,1] >= tau_stage_2):
        patients_stages[i,0] = 3
        patients_stages[i,1] = 2
        
    if (mci_csf_values[i,0] <= abeta_stage_3) & (mci_csf_values[i,1] >= tau_stage_3):
        patients_stages[i,0] = 3
        patients_stages[i,1] = 3

    idx_flat = np.argmin(distances[i, :, :])
    idx_flat_abeta = np.argmin(distances_abeta[i, :, :])
    idx_flat_tau = np.argmin(distances_tau[i, :, :])

    row, column = np.unravel_index(idx_flat, distances[i, :, :].shape)
    row_abeta, column_abeta = np.unravel_index(idx_flat_abeta, distances_abeta[i, :, :].shape) # We consider only rows (abeta estimation)
    row_tau, column_tau = np.unravel_index(idx_flat_tau, distances_tau[i, :, :].shape)         # We consider only columns (tau estimation)

    #print(i, ': ', mci_csf_values[i,:], '; ', patients_stages[i], '; ', row, ' ', column, '; ', row_abeta, ' ', column_tau)
    print("{:3d} : {:>5} {:>5} ; {:>10} ; {:>7} ; {:>7}".format(
        i,
        str(mci_csf_values[i, 0]),
        str(mci_csf_values[i, 1]),
        str(patients_stages[i]),
        f"{row} {column}",
        f"{row_abeta} {column_tau}"
    ))
    """
        print("{:3d} : {:>5} {:>5} ; {:>10} ; {:>7} ; {:>7}".format(
            i,
            str(mci_csf_values[i, 0]),
            str(mci_csf_values[i, 1]),
            str(patients_stages[i]),
            f"{row} {column}",
            f"{row_abeta} {column_tau}"
        ))
    """
    sep_error += np.abs(patients_stages[i,0]-row_abeta)
    sep_error += np.abs(patients_stages[i,1]-column_tau)

    #if (patients_stages[i,0] == row) & (patients_stages[i,1] == column): cont_correclty_id += 1
    #if patients_stages[i,0] == row: cont_correclty_id_abeta += 1
    #if patients_stages[i,1] == column: cont_correclty_id_tau += 1

    #if (patients_stages[i,0] !=0) & (row!=0): cont_correclty_id += 1
    #if (patients_stages[i,0] ==0) & (row==0): cont_correclty_id += 1
    #if (patients_stages[i,1] !=0) & (column!=0): cont_correclty_id += 1
    #if (patients_stages[i,1] ==0) & (column==0): cont_correclty_id += 1
    
    if (patients_stages[i,0] !=0) & (row!=0) & (patients_stages[i,1] !=0) & (column!=0): cont_correclty_id += 1 #case abeta+ tau+
    if (patients_stages[i,0] ==0) & (row==0) & (patients_stages[i,1] !=0) & (column!=0): cont_correclty_id += 1 #case abeta- tau+
    if (patients_stages[i,0] !=0) & (row!=0) & (patients_stages[i,1] ==0) & (column==0): cont_correclty_id += 1 #case abeta+ tau-
    if (patients_stages[i,0] ==0) & (row==0) & (patients_stages[i,1] ==0) & (column==0): cont_correclty_id += 1 #case abeta- tau-

    #if (patients_stages[i,0] ==0) & (row==0): cont_correclty_id_abeta += 1
    #if (patients_stages[i,0] !=0) & (row!=0): cont_correclty_id_abeta += 1
    #if (patients_stages[i,0] ==0) & (column==0): cont_correclty_id_tau += 1
    #if (patients_stages[i,0] !=0) & (column!=0): cont_correclty_id_tau += 1
    
    """
    if (patients_stages[i,0] <) & (row<2) & (patients_stages[i,1] <2) & (column<2): cont_correclty_id += 1 #case abeta- tau-
    if (patients_stages[i,0] >=2) & (row>=2) & (patients_stages[i,1] <2) & (column<2): cont_correclty_id += 1 #case abeta+ tau-
    if (patients_stages[i,0] <2) & (row<2) & (patients_stages[i,1] >=2) & (column>=2): cont_correclty_id += 1 #case abeta- tau+
    if (patients_stages[i,0] >=2) & (row>=2) & (patients_stages[i,1] >=2) & (column>=2): cont_correclty_id += 1 #case abeta+ tau+
    
    if patients_stages[i,0] == row_abeta: sep_correclty_id += 1
    if patients_stages[i,1] == column_tau: sep_correclty_id += 1
    if patients_stages[i,0] == row_abeta: sep_correclty_id_abeta += 1
    if patients_stages[i,1] == column_tau: sep_correclty_id_tau += 1
    """
    if (patients_stages[i,0] !=0) & (row_abeta!=0) & (patients_stages[i,1] !=0) & (column_tau!=0): sep_correclty_id += 1 #case abeta+ tau+
    if (patients_stages[i,0] ==0) & (row_abeta==0) & (patients_stages[i,1] !=0) & (column_tau!=0): sep_correclty_id += 1 #case abeta- tau+
    if (patients_stages[i,0] !=0) & (row_abeta!=0) & (patients_stages[i,1] ==0) & (column_tau==0): sep_correclty_id += 1 #case abeta+ tau-
    if (patients_stages[i,0] ==0) & (row_abeta==0) & (patients_stages[i,1] ==0) & (column_tau==0): sep_correclty_id += 1 #case abeta- tau-
    """
    if (patients_stages[i,0] !=0) & (row_abeta!=0) : sep_correclty_id_abeta += 1 #case abeta+
    if (patients_stages[i,0] ==0) & (row_abeta==0) : sep_correclty_id_abeta += 1 #case abeta-
    if (patients_stages[i,1] !=0) & (column_tau!=0): sep_correclty_id_tau += 1 #case tau+
    if (patients_stages[i,1] ==0) & (column_tau==0): sep_correclty_id_tau += 1 #case tau-
    
    if (patients_stages[i,0] <2) & (row<2): sep_correclty_id += 1 #case abeta-
    if (patients_stages[i,0] >=2) & (row>=2) & (patients_stages[i,1] <2) & (column<2): cont_correclty_id += 1 #case abeta+ tau-
    if (patients_stages[i,0] <2) & (row<2) & (patients_stages[i,1] >=2) & (column>=2): cont_correclty_id += 1 #case abeta- tau+
    if (patients_stages[i,0] >=2) & (row>=2) & (patients_stages[i,1] >=2) & (column>=2): cont_correclty_id += 1 #case abeta+ tau+
    """
    contemp_error += np.abs(patients_stages[i,0]-row)
    contemp_error += np.abs(patients_stages[i,1]-column)

print('correctly identified in contemporary staging', cont_correclty_id)
print('correctly identified in contemporary staging: tau', cont_correclty_id_tau)
print('correctly identified in contemporary staging: abeta', cont_correclty_id_abeta)
print('total error in contemporary staging', contemp_error)
print('correclty identified in two times staging', sep_correclty_id)
print('correctly identified in two times staging: tau', sep_correclty_id_tau)
print('correctly identified in two times staging: abeta', sep_correclty_id_abeta)
print('total error in two times staging', sep_error)

plt.figure(figsize=(10, 6))
plt.scatter(mci_csf_values[:,0], mci_csf_values[:,1])
plt.plot(np.ones(10)*abeta_stage_1, np.linspace(10,160,10))
plt.plot(np.ones(10)*abeta_stage_2, np.linspace(10,160,10))
plt.plot(np.ones(10)*abeta_stage_3, np.linspace(10,160,10))
plt.plot(np.linspace(250,2300,10), np.ones(10)*tau_stage_1)
plt.plot(np.linspace(250,2300,10), np.ones(10)*tau_stage_2)
plt.plot(np.linspace(250,2300,10), np.ones(10)*tau_stage_3)
plt.xlabel('Abeta')
plt.ylabel('p-Tau')
plt.grid()
plt.show()

#print('corretly identified patient: ', correclty_id)
#print('1 stage mismatch: ', one_error)
#print('2 stage mismatch: ', two_error)
#print('3 stage mismatch: ', three_error)
#print('4 stage mismatch: ', four_error)
#print('5 stage mismatch: ', five_error)

########## ovviamente questi sono 3 stage mentre noi ne abbiamo 5, per come abbiamo strutturato i nostri stages
########## la corrispondenda stages_reali vs stages_simulati sarà la seguente:
########## stage_reale 1 --> stage_simulato 1
########## stage_reale 2 --> stage_simulato 2
########## stage_reale 3 --> stage_simulato 3,4,5
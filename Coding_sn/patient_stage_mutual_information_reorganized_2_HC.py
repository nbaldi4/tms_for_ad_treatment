import pandas as pd
from pathlib import Path
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import global_variable_creation as gv
from sklearn.feature_selection import mutual_info_regression

from data_analysis_utils import real_vs_stages_PSD, real_vs_stages_FC, to_common_indices, macro_fc, analyze_regular_data, analyze_reduced_data


# Rimuove il limite di larghezza della colonna (None = nessun limite)
pd.set_option('display.max_colwidth', None)

abeta_stages = 6
tau_stages = 6
patients_number = 19

def hc_metrics(root_path = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data"):
    hc_data = np.load(os.path.join(root_path, r"Patients_highlights\hc_summary_stats_ref_freqs.npz"), allow_pickle=True)
    hc_fc_abs_mean = hc_data['hc_fc_abs_mean']
    hc_psd_rel_pow = hc_data['hc_psd_rel_pow']
    return hc_fc_abs_mean, hc_psd_rel_pow


def compute_metrics():
    root_path = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data"
    file_name = "Stages_16_02_K_stages_"
    stages_metrics = np.zeros((6, 6, 9))

    stages_fc_data = np.load(os.path.join(root_path, os.path.join("Staging_highlights", file_name + "fc.npz")), allow_pickle=True)
    fc_matrices=stages_fc_data['fc_matrices']
    stages_psd_data = np.load(os.path.join(root_path, os.path.join("Staging_highlights", file_name + "psd.npz")), allow_pickle=True)
    psd_arrays=stages_psd_data['psd_arrays']
    psd_freqs=stages_psd_data['psd_freqs']

    for abeta in range(abeta_stages):
        for tau in range(tau_stages):
            stages_metrics[abeta,tau,0] = np.mean(np.abs(fc_matrices[abeta,tau,0,0]))
            stages_metrics[abeta,tau,1] = np.mean(np.abs(fc_matrices[abeta,tau,1,0]))
            stages_metrics[abeta,tau,2] = np.mean(np.abs(fc_matrices[abeta,tau,2,0]))
            stages_metrics[abeta,tau,3] = np.mean(np.abs(fc_matrices[abeta,tau,3,0]))
            stages_metrics[abeta,tau,4] = np.mean(np.abs(fc_matrices[abeta,tau,4,0]))
            freqs = psd_freqs[abeta,tau,0]
            psd_broad = np.sum(psd_arrays[abeta,tau,0][(freqs>= 0.5) & (freqs < 45)])  #broad
            psd_delta = np.sum(psd_arrays[abeta,tau,0][(freqs>= 0.5) & (freqs < 4)])  #delta
            psd_theta = np.sum(psd_arrays[abeta,tau,0][(freqs >= 4) & (freqs < 8)])   #theta
            psd_alpha = np.sum(psd_arrays[abeta,tau,0][(freqs >= 8) & (freqs < 13)])  #alpha
            psd_beta = np.sum(psd_arrays[abeta,tau,0][(freqs >= 13) & (freqs < 30)])  #beta
            stages_metrics[abeta,tau,5] = psd_delta/psd_broad
            stages_metrics[abeta,tau,6] = psd_theta/psd_broad
            stages_metrics[abeta,tau,7] = psd_alpha/psd_broad
            stages_metrics[abeta,tau,8] = psd_beta/psd_broad

    return stages_metrics

def top_values(data, values_to_take):
        top_idx = np.argpartition(data, -values_to_take, axis=3)[:, :, :, -values_to_take:]
        data_top_vals = np.take_along_axis(data, top_idx, axis=3)
        return data_top_vals

def compute_metrics_seeds():
    root_path = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data"
    stages_folder = "Random_seeds_16_02_M_noi_0_05"
    seeds_number = 20
    values_to_take = 13
    assert(values_to_take <= seeds_number)

    output_path_fc = os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Staging_highlights", stages_folder + '_stages_fc.npz')

    output_path_psd = os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Staging_highlights", stages_folder + '_stages_psd.npz')


    loaded_psd_data = np.load(output_path_psd, allow_pickle=True)
    loaded_fc_data = np.load(output_path_fc, allow_pickle=True)
    fc_from_file = loaded_fc_data['fc_matrices']
    psd_from_file = loaded_psd_data['psd_arrays']
    freqs_from_file = loaded_psd_data['psd_freqs']

    stages = 6
    bands = 5
    seeds = seeds_number
    psd_metrics_seeds = np.zeros((stages,stages,bands-1,seeds))
    stages_metrics = np.zeros((stages,stages,9,seeds))

    for abeta in range(stages):
        for tau in range(stages):
            for seed in range(seeds):
                freqs = freqs_from_file[abeta,tau,seed]
                psd_broad = np.sum(psd_from_file[abeta,tau,seed][(freqs>= 0.5) & (freqs < 45)])  #broad
                psd_delta = np.sum(psd_from_file[abeta,tau,seed][(freqs>= 0.5) & (freqs < 4)])  #delta
                psd_theta = np.sum(psd_from_file[abeta,tau,seed][(freqs >= 4) & (freqs < 8)])   #theta
                psd_alpha = np.sum(psd_from_file[abeta,tau,seed][(freqs >= 8) & (freqs < 13)])  #alpha
                psd_beta = np.sum(psd_from_file[abeta,tau,seed][(freqs >= 13) & (freqs < 30)])  #beta
                
                psd_metrics_seeds[abeta,tau,0,seed] = psd_delta/psd_broad
                psd_metrics_seeds[abeta,tau,1,seed] = psd_theta/psd_broad
                psd_metrics_seeds[abeta,tau,2,seed] = psd_alpha/psd_broad
                psd_metrics_seeds[abeta,tau,3,seed] = psd_beta/psd_broad

    fc_metrics_seeds = np.zeros((stages, stages, bands, seeds))
    for abeta in range(stages):
        for tau in range(stages):
            for band in range(bands):
                for seed in range(seeds):
                    fc_metrics_seeds[abeta, tau, band, seed] = np.mean(np.abs(fc_from_file[abeta, tau, band, seed]))

    stages_metrics[:,:,:5,:] = fc_metrics_seeds
    stages_metrics[:,:,5:,:] = psd_metrics_seeds

    print("abeta 0, tau 0, fc broad band, seed 0")
    print(stages_metrics[0,0,0,0])
    print("abeta 3, tau 2, fc delta band, seed 1")
    print(stages_metrics[3,2,1,1])
    print("abeta 4, tau 5, fc alpha band, seed 10")
    print(stages_metrics[4,5,3,10])
    print("abeta 0, tau 0, psd beta/broad, seed 0")
    print(stages_metrics[0,0,8,0])
    print("abeta 3, tau 2, psd delta/broad, seed 1")
    print(stages_metrics[3,2,5,1])
    print("abeta 4, tau 5, psd alpha/broad, seed 10")
    print(stages_metrics[4,5,7,10])


    print(stages_metrics.shape)
    top_idx = np.argpartition(stages_metrics, -values_to_take, axis=3)[:, :, :, -values_to_take:]
    stages_metrics_top_vals = np.take_along_axis(stages_metrics, top_idx, axis=3)

    stages_metrics_top = np.zeros((abeta_stages, tau_stages, 9))

    for abeta in range(stages):
        for tau in range(stages):
            for band in range(9):
                stages_metrics_top[abeta, tau, band] = np.median(stages_metrics_top_vals[abeta, tau, band, :])
    """
    print("prima di ordinare")
    print(stages_metrics)
    print("dopo l'ordinamento")
    print(stages_metrics_top_vals)
    print("mediana")
    print(stages_metrics_top)

    for abeta in range(stages):
        for tau in range(stages):
            if (abeta == tau) | (abeta == 0) | (tau == 0):
                print("abeta ", abeta, "; tau ", tau, "; fc broad")
                print(stages_metrics_top[abeta, tau, 0])
                print("abeta ", abeta, "; tau ", tau, "; fc delta")
                print(stages_metrics_top[abeta, tau, 1])
                print("abeta ", abeta, "; tau ", tau, "; fc beta")
                print(stages_metrics_top[abeta, tau, 4])
                print("abeta ", abeta, "; tau ", tau, "; psd theta/broad")
                print(stages_metrics_top[abeta, tau, 6])
                print("abeta ", abeta, "; tau ", tau, "; psd beta/broad")
                print(stages_metrics_top[abeta, tau, 8])
    """
    return stages_metrics_top

def compute_metrics_seeds_tavg():
    root_path = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data"
    seeds_number = 40
    values_to_take = 39
    assert(values_to_take <= seeds_number)

    output_path_fc = os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Staging_highlights", "Rnd_seeds_M_FC_pears_tavg" + '_stages_fc.npz')

    output_path_psd = os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Staging_highlights", "Rnd_seeds_M_FC_pears_tavg" + '_stages_psd.npz')


    loaded_psd_data = np.load(output_path_psd, allow_pickle=True)
    loaded_fc_data = np.load(output_path_fc, allow_pickle=True)
    fc_from_file = loaded_fc_data['fc_matrices']
    psd_from_file = loaded_psd_data['psd_arrays']
    freqs_from_file = loaded_psd_data['psd_freqs']

    stages = 6
    bands = 5
    seeds = seeds_number
    psd_metrics_seeds = np.zeros((stages,stages,bands-1,seeds))
    stages_metrics = np.zeros((stages,stages,9,seeds))

    for abeta in range(stages):
        for tau in range(stages):
            for seed in range(seeds):
                freqs = freqs_from_file[abeta,tau,seed]
                psd_broad = np.sum(psd_from_file[abeta,tau,seed][(freqs>= 0.5) & (freqs < 45)])  #broad
                psd_delta = np.sum(psd_from_file[abeta,tau,seed][(freqs>= 0.5) & (freqs < 4)])  #delta
                psd_theta = np.sum(psd_from_file[abeta,tau,seed][(freqs >= 4) & (freqs < 8)])   #theta
                psd_alpha = np.sum(psd_from_file[abeta,tau,seed][(freqs >= 8) & (freqs < 13)])  #alpha
                psd_beta = np.sum(psd_from_file[abeta,tau,seed][(freqs >= 13) & (freqs < 30)])  #beta
                
                psd_metrics_seeds[abeta,tau,0,seed] = psd_delta/psd_broad
                psd_metrics_seeds[abeta,tau,1,seed] = psd_theta/psd_broad
                psd_metrics_seeds[abeta,tau,2,seed] = psd_alpha/psd_broad
                psd_metrics_seeds[abeta,tau,3,seed] = psd_beta/psd_broad

    fc_metrics_seeds = np.zeros((stages, stages, bands, seeds))
    for abeta in range(stages):
        for tau in range(stages):
            for band in range(bands):
                for seed in range(seeds):
                    fc_metrics_seeds[abeta, tau, band, seed] = np.mean(np.abs(fc_from_file[abeta, tau, band, seed]))

    stages_metrics[:,:,:5,:] = fc_metrics_seeds
    stages_metrics[:,:,5:,:] = psd_metrics_seeds

    print("abeta 0, tau 0, fc broad band, seed 0")
    print(stages_metrics[0,0,0,0])
    print("abeta 3, tau 2, fc delta band, seed 1")
    print(stages_metrics[3,2,1,1])
    print("abeta 4, tau 5, fc alpha band, seed 4")
    print(stages_metrics[4,5,3,4])
    print("abeta 0, tau 0, psd beta/broad, seed 0")
    print(stages_metrics[0,0,8,0])
    print("abeta 3, tau 2, psd delta/broad, seed 1")
    print(stages_metrics[3,2,5,1])
    print("abeta 4, tau 5, psd alpha/broad, seed 4")
    print(stages_metrics[4,5,7,4])

    top_idx = np.argpartition(stages_metrics, values_to_take, axis=3)[:, :, :, :values_to_take]
    stages_metrics_top_vals = np.take_along_axis(stages_metrics, top_idx, axis=3)

    stages_metrics_top = np.zeros((abeta_stages, tau_stages, 9))

    for abeta in range(stages):
        for tau in range(stages):
            for band in [0,1,2,3,4]:
                stages_metrics_top[abeta, tau, band] = np.median(stages_metrics[abeta, tau, band, :])
            for band in [5,6,7,8]:
                stages_metrics_top[abeta, tau, band] = np.median(stages_metrics_top_vals[abeta, tau, band, :])
    """
    print("prima di ordinare")
    print(stages_metrics[0,5,1,:])
    print("dopo l'ordinamento")
    print(stages_metrics_top_vals[0,5,1,:])
    for abeta in range(stages):
        for tau in range(stages):
            if (abeta == tau) | (abeta == 0) | (tau == 0):
                print("abeta ", abeta, "; tau ", tau, "; fc broad")
                print(stages_metrics_top[abeta, tau, 0])
                print("abeta ", abeta, "; tau ", tau, "; fc delta")
                print(stages_metrics_top[abeta, tau, 1])
                print("abeta ", abeta, "; tau ", tau, "; fc beta")
                print(stages_metrics_top[abeta, tau, 4])
                print("abeta ", abeta, "; tau ", tau, "; psd theta/broad")
                print(stages_metrics_top[abeta, tau, 6])
                print("abeta ", abeta, "; tau ", tau, "; psd beta/broad")
                print(stages_metrics_top[abeta, tau, 8])
    """

    return stages_metrics_top

hc_fc_abs_mean, hc_psd_rel_pow = hc_metrics()

print(hc_fc_abs_mean)
print(hc_psd_rel_pow)

stages_metrics = compute_metrics_seeds()

# We decide not to include regular abs theta in metrics used because we noted a rise in algorithm performance when it was not included
def contemporary_staging(p_idx):
    distance = np.zeros((abeta_stages, tau_stages))
    for abeta_idx in range(abeta_stages):
        for tau_idx in range(tau_stages):
            distance[abeta_idx, tau_idx] = (np.abs(hc_psd_rel_pow[p_idx, 1] - stages_metrics[abeta_idx, tau_idx, 6]) + # theta/broad
                                            np.abs(hc_psd_rel_pow[p_idx, 3] - stages_metrics[abeta_idx, tau_idx, 8]) + # beta/broad
                                            np.abs(hc_fc_abs_mean[p_idx, 0] - stages_metrics[abeta_idx, tau_idx, 0]) + # regular abs broad
                                            np.abs(hc_fc_abs_mean[p_idx, 1] - stages_metrics[abeta_idx, tau_idx, 1]) + # regular abs delta
                                            #np.abs(mci_fc_abs_mean[p_idx, 2] - stages_metrics[abeta_idx, tau_idx, 2]) + # regular abs theta
                                            np.abs(hc_fc_abs_mean[p_idx, 4] - stages_metrics[abeta_idx, tau_idx, 4])   # regular abs beta
                                            )
            
    return distance
            
def two_times_staging(p_idx):
    distance_tau = np.zeros((abeta_stages, tau_stages))
    for abeta_idx in range(abeta_stages):
        for tau_idx in range(tau_stages):
            distance_tau[abeta_idx, tau_idx] = (np.abs(hc_psd_rel_pow[p_idx, 3] - stages_metrics[abeta_idx, tau_idx, 8]) + # beta/broad
                                                np.abs(hc_fc_abs_mean[p_idx, 1] - stages_metrics[abeta_idx, tau_idx, 1]) + # regular abs delta
                                                np.abs(hc_fc_abs_mean[p_idx, 4] - stages_metrics[abeta_idx, tau_idx, 4])   # regular abs beta
                                                )
                
    distance_abeta = np.zeros((abeta_stages, tau_stages))
    for abeta_idx in range(abeta_stages):
        for tau_idx in range(tau_stages):
            distance_abeta[abeta_idx, tau_idx] = (np.abs(hc_psd_rel_pow[p_idx, 1] - stages_metrics[abeta_idx, tau_idx, 6]) + # theta/broad
                                                  np.abs(hc_psd_rel_pow[p_idx, 3] - stages_metrics[abeta_idx, tau_idx, 8]) + # beta/broad
                                                  np.abs(hc_fc_abs_mean[p_idx, 0] - stages_metrics[abeta_idx, tau_idx, 0]) + # regular abs broad
                                                  #np.abs(mci_fc_abs_mean[p_idx, 2] - stages_metrics[abeta_idx, tau_idx, 2]) + # regular abs theta
                                                  np.abs(hc_fc_abs_mean[p_idx, 1] - stages_metrics[abeta_idx, tau_idx, 1])   # regular abs delta
                                                  )
            
    return distance_abeta, distance_tau
            


######## ora determino lo stage dei pazienti in funzione dei protein levels determinati a partire dall'articolo"Clinical and biomarker trajectories in sporadic Alzheimer’s
######## disease: A longitudinal study" di Alzheimer's and Dementia
#CSF abeta level at stage I:  0.2318534187814261
#CSF abeta level at stage II:  0.10060892803728635
#CSF abeta level at stage III:  0.05000000000000001
#CSF tau level at stage I:  0.5553246833941619
#CSF tau level at stage II:  0.8543703680827017
#CSF tau level at stage III:  0.9500000000000001
######## per fare le proporzioni con i nostri livelli di proteine abbiamo soltanto il livello soglia, che supponiamo sia quello di cutoff
######## p-tau >= 52 mg/L che corrisponde a 30.29 (inferred)
######## abeta <= 813 mg/L che corrisponde a 192 (labeled in the article); la percentuale di decrescita della proteina alla quale è fissato il cutoff è 0.20618514077810396
######## adesso dobbiamo risalire ai valori corrispondenti di abeta e tau nei 3 stages
######## partiamo con i valori massimo di abeta e  tau 
######## il valore massimo di abeta sarà dato dalla proporzione x:813 = 241.87:192 --> x = 1024.17
######## di conseguenza possiamo calcolare il valore minimo di abeta x:1024.17 = 130.53:241.87 --> x = 552.71
######## allo stesso modo il valore massimo di tau sarà dato da x:52 = 47.07:30.29 --> x = 80.81
######## quindi i delta del nostro modello saranno: abeta 1024.17 - 552.71 = 471.46
########                                            tau 80.81 - 52 = 28.81

#########con questa divisione in stage la performance dell'algoritmo sale a 10 match e 4 mismatch di 1 (su 30 confronti totali)

def alternative_patient_stage_identification(three_stage_division = True,
                                 two_stage_division = True,
                                 patient_distribution = True,
                                 patients_number = patients_number
                                 ):
    patients_stages = np.zeros((patients_number, 2)) # tau stage, abeta stage

    three_stage_cont_match = 0
    two_stage_cont_match = 0
    three_stage_sep_match = 0
    two_stage_sep_match = 0

    for i in range(patients_number):
        dist_cont = contemporary_staging(i)
        idx_flat = np.argmin(dist_cont)
        dist_ab, dist_ta = two_times_staging(i)
        idx_flat_abeta = np.argmin(dist_ab)
        idx_flat_tau = np.argmin(dist_ta)
        #if i ==41:
        #    print(dist_cont)
        #    print(dist_ab)
        #    print(dist_ta)

        row, column = np.unravel_index(idx_flat, dist_cont.shape)
        row_abeta, column_abeta = np.unravel_index(idx_flat_abeta, dist_ab.shape) # We consider only rows (abeta estimation)
        row_tau, column_tau = np.unravel_index(idx_flat_tau, dist_ta.shape)       # We consider only columns (tau estimation)

        #print(i, ': ', mci_csf_values[i,:], '; ', patients_stages[i], '; ', row, ' ', column, '; ', row_abeta, ' ', column_tau)
        print("{:3d} : {:>10} ; {:>7} ; {:>7}".format(
            i,
            str(patients_stages[i]),
            f"{row} {column}",
            f"{row_abeta} {column_tau}"
        ))

        if (row == 0) | (row == 1): row_eff = 0
        if (row == 2): row_eff = 1
        if (row == 3): row_eff = 2
        if (row == 4) | (row == 5): row_eff = 3

        if (column == 0) | (column == 1): column_eff = 0
        if (column == 2): column_eff = 1
        if (column == 3): column_eff = 2
        if (column == 4) | (column == 5): column_eff = 3

        if (row_abeta == 0) | (row_abeta == 1): row_abeta_eff = 0
        if (row_abeta == 2): row_abeta_eff = 1
        if (row_abeta == 3): row_abeta_eff = 2
        if (row_abeta == 4) | (row_abeta == 5): row_abeta_eff = 3
        
        if (column_tau == 0) | (column_tau == 1): column_tau_eff = 0
        if (column_tau == 2): column_tau_eff = 1
        if (column_tau == 3): column_tau_eff = 2
        if (column_tau == 4) | (column_tau == 5): column_tau_eff = 3

        if three_stage_division == True:

            if (patients_stages[i,0] == row_eff) & (patients_stages[i,1] == column_eff): three_stage_cont_match += 1
    
            if (patients_stages[i,0] == row_abeta_eff) & (patients_stages[i,1] == column_tau_eff): three_stage_sep_match += 1

        if two_stage_division == True:
            if (patients_stages[i,0] !=0) & (row_eff!=0) & (patients_stages[i,1] !=0) & (column_eff!=0): two_stage_cont_match += 1 #case abeta+ tau+
            if (patients_stages[i,0] ==0) & (row_eff==0) & (patients_stages[i,1] !=0) & (column_eff!=0): two_stage_cont_match += 1 #case abeta- tau+
            if (patients_stages[i,0] !=0) & (row_eff!=0) & (patients_stages[i,1] ==0) & (column_eff==0): two_stage_cont_match += 1 #case abeta+ tau-
            if (patients_stages[i,0] ==0) & (row_eff==0) & (patients_stages[i,1] ==0) & (column_eff==0): two_stage_cont_match += 1 #case abeta- tau-
    
            if (patients_stages[i,0] !=0) & (row_abeta_eff!=0) & (patients_stages[i,1] !=0) & (column_tau_eff!=0): two_stage_sep_match += 1 #case abeta+ tau+
            if (patients_stages[i,0] ==0) & (row_abeta_eff==0) & (patients_stages[i,1] !=0) & (column_tau_eff!=0): two_stage_sep_match += 1 #case abeta- tau+
            if (patients_stages[i,0] !=0) & (row_abeta_eff!=0) & (patients_stages[i,1] ==0) & (column_tau_eff==0): two_stage_sep_match += 1 #case abeta+ tau-
            if (patients_stages[i,0] ==0) & (row_abeta_eff==0) & (patients_stages[i,1] ==0) & (column_tau_eff==0): two_stage_sep_match += 1 #case abeta- tau-        

    if three_stage_division == True:
        print('correctly identified in contemporary staging using three stage division', three_stage_cont_match)
        print('correclty identified in two times staging using three stage division', three_stage_sep_match)
    if two_stage_division == True:
        print('correctly identified in contemporary staging using two stage division', two_stage_cont_match)
        print('correclty identified in two times staging using two stage division', two_stage_sep_match)

#patient_stage_identification()
alternative_patient_stage_identification(patient_distribution=False)
#mutual_info()
import pandas as pd
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import global_variable_creation as gv
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr

from data_analysis_utils import real_vs_stages_PSD, real_vs_stages_FC, to_common_indices, macro_fc, analyze_regular_data, analyze_reduced_data


# Rimuove il limite di larghezza della colonna (None = nessun limite)
pd.set_option('display.max_colwidth', None)

#########i parametri che cambiano sono: J, b e w per quanto riguarda tau
######################################: a e b per quanto riguarda beta

def parameters_value(abeta_stage, tau_stage):
    a=0.0728044836955774
    b=0.06293568322880852
    weights = np.load(gv.struct_path)[tau_stage, :, :]
    weights /= 3 
    # Model parameters (based on Jansen–Rit equations)
    J_value = 128
    J_stages = np.ones((76))
    tau_stage_1 = np.array([ #2 areas
        24, 62   # PHC
    ])

    tau_stage_2 = np.array([ #8 areas
        9, 47,       # Temporal allocortex / hippocampal limbic III -> HC
        31, 69,      # TCI
        30, 68,      # TCC
        34, 72       # TCV
    ])
    tau_stage_3 = np.array([ #16 areas
        18, 56,   # PFCDL
        17, 55,   # PFCCL
        19, 57,   # PFCDM
        12, 50,   # M1
        28, 66,   # S1
        29, 67,   # S2
        35, 73,   # V1 
        36, 74    # V2
    ])

    Abeta_stage_1 = [ #12 areas
        22, 60,   # PFCPOL (prefrontal polar)
        21, 59,   # PFCORB (orbitofrontal)
        31, 69,   # TCI (inferior temporal)
        30, 68,   # TCC (central temporal)
        34, 72,   # TCV (ventral temporal)
        32, 70    # TCPOL (temporal pole)
    ]
    Abeta_stage_2 = [ #14 areas
        9, 47,      # Hippocampus (HC)
        24, 62,     # Parahippocampal gyrus (PHC)
        2, 40,      # Amygdala (AMYG)
        10, 48,     # Insula (IA) = anterior + posterior
        20, 58,     # Subgenual cingulum (PFCM)
        5, 43,      # Retrosplenial cingulum (CCR)
        23, 61      # Ventrolateral prefrontal cortex (PFCVL)
    ]
    Abeta_stage_3 = [ #50 areas
        0,  1,
        3,  4,
        6,  7,
        8, 11,
        12, 13,
        14, 15,
        16, 17,
        18, 19,
        25, 26,
        27, 28,
        29, 33,
        35, 36,
        37, 38,
        39, 41,
        42, 44,
        45, 46,
        49, 50,
        51, 52,
        53, 54,
        55, 56,
        57, 63,
        64, 65,
        66, 67,
        71, 73,
        74, 75
    ]

    degeneration = [0, 0, 0, 0.28, 0.78, 0.95, 0.95, 0.95]
    degeneration_2_tau = [0, 0, 0, 0.28, 0.78, 0.95, 0.95, 0.95]
    degeneration_2_abeta = [0, 0, 0, 0.6116, 0.8664, 0.95, 0.95, 0.95]

    a_delta = 3.65/3.25
    b_delta = 0.3133/0.0593

    a_array = np.ones((76,1))*a ########################################################################
    b_array = np.ones((76,1))*b*(1+b_delta*0.01*tau_stage)

    J_delta = 1

    for i in tau_stage_1:
        J_stages[i] = 1 - degeneration_2_tau[tau_stage + 2]*J_delta
        b_array[i] = b*(1+b_delta*0.5*degeneration_2_tau[tau_stage + 2])
    
    for i in tau_stage_2:
        J_stages[i] = 1 - degeneration_2_tau[tau_stage + 1]*J_delta
        b_array[i] = b*(1+b_delta*0.5*degeneration_2_tau[tau_stage + 1])

    for i in tau_stage_3:
        J_stages[i] = 1 - degeneration_2_tau[tau_stage]*J_delta
        b_array[i] = b*(1+b_delta*0.5*degeneration_2_tau[tau_stage])

    for i in Abeta_stage_1:
        a_array[i] = a*(1+a_delta*degeneration_2_abeta[abeta_stage + 2])
        b_array[i] = b_array[i] + b*b_delta*0.5*degeneration_2_abeta[abeta_stage+ 2]

    for i in Abeta_stage_2:
        a_array[i] = a*(1+a_delta*degeneration_2_abeta[abeta_stage + 1])
        b_array[i] = b_array[i] + b*b_delta*0.5*degeneration_2_abeta[abeta_stage + 1]

    for i in Abeta_stage_3:
        a_array[i] = a*(1+a_delta*degeneration_2_abeta[abeta_stage])
        b_array[i] = b_array[i] + b*b_delta*0.5*degeneration_2_abeta[abeta_stage]

    a_mean = np.mean(a_array)
    b_mean = np.mean(b_array)
    J_mean = J_value*np.mean(J_stages)
    weights_mean = np.mean(weights)

    return a_mean, b_mean, J_mean, weights_mean

tau_stages_number = 6
abeta_stages_number = 6
stages_parameters = np.array((tau_stages_number, abeta_stages_number, 4)) #4 are the JR parameters which change with the stages

patient_data = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\MCI_data\patients_common_indices_metrics.npz")
patient_psd_metrics = patient_data['psd_metrics'] #num_patients, metrics [delta/broad; alpha/broad]
patient_fc_metrics = patient_data['fc_metrics'] #num_patients, metrics [regular abs delta; regular abs theta; regular abs alpha]
patient_number = len(patient_psd_metrics[:,0])
patient_parameters = np.zeros((patient_number, 4))

patient_metrics = np.zeros((patient_number, len(patient_psd_metrics[0,:]) + len(patient_fc_metrics[0,:])))
for i in range(patient_number):
    patient_metrics[i,0] = patient_psd_metrics[i,0] #delta/broad
    patient_metrics[i,1] = patient_psd_metrics[i,1] #alpha/broad
    patient_metrics[i,2] = patient_psd_metrics[i,2] #theta/broad
    patient_metrics[i,3] = patient_psd_metrics[i,3] #beta/broad
    patient_metrics[i,4] = patient_fc_metrics[i,0] #regular abs delta
    patient_metrics[i,5] = patient_fc_metrics[i,1] #regular abs theta
    patient_metrics[i,6] = patient_fc_metrics[i,2] #regular abs alpha
    patient_metrics[i,7] = patient_fc_metrics[i,2] #regular abs beta
    patient_metrics[i,8] = patient_fc_metrics[i,2] #regular abs broad

abeta_level = np.loadtxt(r"C:\Users\User\OneDrive - University of Pisa\Desktop\CSF_abeta.csv", skiprows=1)
tau_level = np.loadtxt(r"C:\Users\User\OneDrive - University of Pisa\Desktop\CSF_p_tau.csv", skiprows=1)
prot_lev = np.column_stack((tau_level, abeta_level))

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
nomi_delle_mie_metriche = ['delta/broad', 'theta/broad', 'alpha/broad', 'beta/broad',
                           'regular abs broad','regular abs delta', 'regular abs theta', 'regular abs alpha', 'regular abs beta'
                          ]

#DataFrame convertion
patient_metrics_df = pd.DataFrame(patient_metrics, columns=nomi_delle_mie_metriche)
mi_abeta = calcola_ranking_mi(patient_metrics_df, abeta_level[:len(patient_psd_metrics[:,0])], "Abeta CSF")
print(mi_abeta) 
#le metriche con la mi più alta per abeta sono: 
#regular abs theta  0.195262
#regular abs alpha  0.061525
#alpha/broad  0.048550

mi_tau = calcola_ranking_mi(patient_metrics_df, tau_level[:len(patient_psd_metrics[:,0])], "Tau CSF")
print(mi_tau)
#le metriche con la mi più alta per tau sono:
#regular abs delta  0.223435
#delta/broad  0.018876

# Analisi per Tau
#mi_tau = calcola_ranking_mi(df_metriche, df_csf['Tau_CSF'], "Tau CSF")

stages_data = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\Staging_del_25_01_(new_staging)\stage_metrics_del_25_01_(new_staging).npz")
stages_psd_metrics = stages_data['psd_metrics'] #tau stage, abeta stage, metrics [delta/broad; alpha/broad]
stages_fc_metrics = stages_data['fc_metrics'] #tau stage, abeta stage, metrics [regular abs delta; regular abs theta; regular abs alpha]
print(stages_fc_metrics.shape)
distances = np.zeros((len(patient_fc_metrics[:,0]), len(stages_fc_metrics[:,0,0]), len(stages_fc_metrics[0,:,0]))) #patients, tau, abeta
for p_idx in range(len(patient_fc_metrics[:,0])):
    for tau_idx in range(len(stages_fc_metrics[:,0,0])):
        for abeta_idx in range(len(stages_fc_metrics[0,:,0])):
            distances[p_idx, tau_idx, abeta_idx] = (np.abs(patient_psd_metrics[p_idx, 0] - stages_psd_metrics[tau_idx, abeta_idx, 0]) + 
                                                   np.abs(patient_psd_metrics[p_idx, 1] - stages_psd_metrics[tau_idx, abeta_idx, 1]) +
                                                   np.abs(patient_fc_metrics[p_idx, 0] - stages_fc_metrics[tau_idx, abeta_idx, 0]) +
                                                   np.abs(patient_fc_metrics[p_idx, 2] - stages_fc_metrics[tau_idx, abeta_idx, 2]) 
                                                   )

patients_stages = np.zeros((patient_number, 2)) # tau stage, abeta stage
inferred_patients_stages = np.zeros((patient_number, 2)) # tau stage, abeta stage
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
abeta_stage_4 = 552.71 - abeta_delta*0.05000000000000001
abeta_stage_5 = 552.71 - abeta_delta*0.10060892803728635

print(abeta_stage_1)
print(abeta_stage_2)
print(abeta_stage_3)

tau_delta = 80.81 - 28.81
tau_stage_1 = 28.81 + tau_delta*0.5553246833941619
tau_stage_2 = 28.81 + tau_delta*0.8543703680827017
tau_stage_3 = 28.81 + tau_delta*0.9500000000000001
tau_stage_4 = 28.81 + tau_delta*1.5053246833941619
tau_stage_5 = 28.81 + tau_delta*1.8043703680827018

print(tau_stage_1)
print(tau_stage_2)
print(tau_stage_3)
#########con questa divisione in stage la performance dell'algoritmo sale a 10 match e 4 mismatch di 1 (su 30 confronti totali)

"""
for i in range(patient_number):
    if (prot_lev[i,1] == 0) & (prot_lev[i,0] == 0):
        patients_stages[i,1] = np.nan
        patients_stages[i,0] = np.nan
    #prot_lev[i,0] < abeta_stage_1
    if (prot_lev[i,1] < abeta_stage_1) & (prot_lev[i,0] < tau_stage_1):
        patients_stages[i,1] = 0
        patients_stages[i,0] = 0

    if (prot_lev[i,1] < abeta_stage_1) & (tau_stage_2 > prot_lev[i,0] >= tau_stage_1):
        patients_stages[i,1] = 3
        patients_stages[i,0] = 3

    if (prot_lev[i,1] < abeta_stage_1) & (tau_stage_3 > prot_lev[i,0] >= tau_stage_2):
        patients_stages[i,1] = 4
        patients_stages[i,0] = 3
       
    if (prot_lev[i,1] < abeta_stage_1) & (prot_lev[i,0] >= tau_stage_3):
        patients_stages[i,1] = 5
        patients_stages[i,0] = 3

    #abeta_stage_2 > prot_lev[i,0] >= abeta_stage_1
    if (abeta_stage_2 > prot_lev[i,1] >= abeta_stage_1) & (prot_lev[i,0] < tau_stage_1):
        patients_stages[i,1] = 2
        patients_stages[i,0] = 0

    if (abeta_stage_2 > prot_lev[i,1] >= abeta_stage_1) & (tau_stage_2 > prot_lev[i,0] >= tau_stage_1):
        patients_stages[i,1] = 3
        patients_stages[i,0] = 1

    if (abeta_stage_2 > prot_lev[i,1] >= abeta_stage_1) & (tau_stage_3 > prot_lev[i,0] >= tau_stage_2):
        patients_stages[i,1] = 4
        patients_stages[i,0] = 2
        
    if (abeta_stage_2 > prot_lev[i,1] >= abeta_stage_1) & (prot_lev[i,0] >= tau_stage_3):
        patients_stages[i,1] = 5
        patients_stages[i,0] = 3

    #abeta_stage_3 > prot_lev[i,0] >= abeta_stage_2
    if (abeta_stage_3 > prot_lev[i,1] >= abeta_stage_2) & (prot_lev[i,0] < tau_stage_1):
        patients_stages[i,1] = 2
        patients_stages[i,0] = 0

    if (abeta_stage_3 > prot_lev[i,1] >= abeta_stage_2) & (tau_stage_2 > prot_lev[i,0] >= tau_stage_1):
        patients_stages[i,1] = 3
        patients_stages[i,0] = 1

    if (abeta_stage_3 > prot_lev[i,1] >= abeta_stage_2) & (tau_stage_3 > prot_lev[i,0] >= tau_stage_2):
        patients_stages[i,1] = 4
        patients_stages[i,0] = 2
        
    if (abeta_stage_3 > prot_lev[i,1] >= abeta_stage_2) & (prot_lev[i,0] >= tau_stage_3):
        patients_stages[i,1] = 5
        patients_stages[i,0] = 3

    #prot_lev[i,0] >= abeta_stage_3
    if (prot_lev[i,1] >= abeta_stage_3) & (prot_lev[i,0] < tau_stage_1):
        patients_stages[i,1] = 3
        patients_stages[i,0] = 0

    if (prot_lev[i,1] >= abeta_stage_3) & (tau_stage_2 > prot_lev[i,0] >= tau_stage_1):
        patients_stages[i,1] = 3
        patients_stages[i,0] = 1

    if (prot_lev[i,1] >= abeta_stage_3) & (tau_stage_3 > prot_lev[i,0] >= tau_stage_2):
        patients_stages[i,1] = 3
        patients_stages[i,0] = 2
        
    if (prot_lev[i,1] >= abeta_stage_3) & (prot_lev[i,0] >= tau_stage_3):
        patients_stages[i,1] = 3
        patients_stages[i,0] = 3

    idx_flat = np.argmin(distances[i, :, :])

    # 2. Convertilo in coordinate (riga, colonna)
    tau, abeta = np.unravel_index(idx_flat, distances[i, :, :].shape)
    inferred_patients_stages[i,0] = tau
    inferred_patients_stages[i,1] = abeta
    print(i, ': ', prot_lev[i], '; ', patients_stages[i], '; ', tau, ' ', abeta)
"""

for i in range(patient_number):
    if (prot_lev[i,0] == 0) & (prot_lev[i,1] == 0):
        patients_stages[i,0] = np.nan
        patients_stages[i,1] = np.nan
    #prot_lev[i,0] < abeta_stage_1
    if (prot_lev[i,0] > abeta_stage_1):
        patients_stages[i,0] = 0
        
    if (abeta_stage_1 > prot_lev[i,0] > abeta_stage_2):
        patients_stages[i,0] = 1
        
    if (abeta_stage_2 > prot_lev[i,0] > abeta_stage_3):
        patients_stages[i,0] = 2
        
    if (abeta_stage_3 > prot_lev[i,0] > abeta_stage_4):
        patients_stages[i,0] = 3
        
    if (abeta_stage_4 > prot_lev[i,0] > abeta_stage_5):
        patients_stages[i,0] = 4
        
    if (abeta_stage_5 > prot_lev[i,0]):
        patients_stages[i,0] = 5

    if (prot_lev[i,1] < tau_stage_1):
        patients_stages[i,1] = 0

    if (abeta_stage_1 < prot_lev[i,1] < abeta_stage_2):
        patients_stages[i,1] = 1
        
    if (abeta_stage_2 < prot_lev[i,1] < abeta_stage_3):
        patients_stages[i,1] = 2
        
    if (abeta_stage_3 < prot_lev[i,1] < abeta_stage_4):
        patients_stages[i,1] = 3
        
    if (abeta_stage_4 < prot_lev[i,1] < abeta_stage_5):
        patients_stages[i,1] = 4
        
    if (abeta_stage_5 < prot_lev[i,1]):
        patients_stages[i,1] = 5

    idx_flat = np.argmin(distances[i, :, :])

    # 2. Convertilo in coordinate (riga, colonna)
    tau, abeta = np.unravel_index(idx_flat, distances[i, :, :].shape)
    inferred_patients_stages[i,0] = tau
    inferred_patients_stages[i,1] = abeta
    print(i, ': ', prot_lev[i], '; ', patients_stages[i], '; ', tau, ' ', abeta)


for i in range(patient_number):
    if not np.isnan(patients_stages[i]).any():
        patient_parameters[i,:] = parameters_value(tau_stage=int(inferred_patients_stages[i,0]), abeta_stage=int(inferred_patients_stages[i,1]))

print(patients_stages.shape)
print(prot_lev[:patient_number].shape)
valid = (
    ~np.isnan(patients_stages).any(axis=1)
    & ~np.isnan(prot_lev[:patient_number]).any(axis=1) 
    & (prot_lev[:patient_number, 0] > 0)
    & (prot_lev[:patient_number, 1] > 0)
)
prot_lev_sub = prot_lev[:patient_number]
tau = prot_lev_sub[valid, 0]
abeta = prot_lev_sub[valid, 1]
params = patient_parameters[valid, :]
np.set_printoptions(precision=3, suppress=True)
#print('a_mean,  b_mean,  J_mean,  weights,  tau,  abeta')
#print(np.column_stack((params, inferred_patients_stages[valid,:])))
nomi_parametri = ['a_mean', 'b_mean', 'J_mean', 'weights_mean']
print("\n--- Correlazione parametri vs Tau ---")
for i, name in enumerate(nomi_parametri):
    rho, p = spearmanr(tau, params[:, i])
    #if i == 0:print(tau)
    #if i == 0:print(params[:,i])
    mi_tau = mutual_info_regression(
        tau.reshape(-1,1),
        params[:, i],
        random_state=0
        )
    print(f'tau information on parameter {name}: {mi_tau}')

    if gv.save_plots == True:
        plt.scatter(tau, params[:, i], label=name)
        plt.title(f'Correlation: {rho:.3f}, p-value: {p:.3f}')
        plt.xlabel('CSF Tau')
        plt.ylabel(nomi_parametri[i])
        plt.grid()
        plt.legend()
        plt.savefig(r"C:\Users\User\OneDrive - University of Pisa\Desktop\Correlation_scatterplots\tau_"+name)
        plt.close()
    #for j in range(len(tau)):
    #    print(tau[j], params[j,i],)
    print(f"{name:25s}  rho={rho:+.3f}  p={p:.3e}")
print("\n--- Correlazione parametri vs Abeta ---")
for i, name in enumerate(nomi_parametri):
    rho, p = spearmanr(abeta, params[:, i])
    mi_abeta = mutual_info_regression(
        abeta.reshape(-1,1),
        params[:, i],
        random_state=0
        )
    print(f'tau information on parameter {name}: {mi_abeta}')

    if gv.save_plots == True:
        plt.scatter(abeta, params[:, i], label=name)
        plt.title(f'Correlation: {rho:.3f}, p-value: {p:.3f}')
        plt.xlabel('CSF Abeta')
        plt.ylabel(nomi_parametri[i])
        plt.grid()
        plt.legend()
        plt.savefig(r"C:\Users\User\OneDrive - University of Pisa\Desktop\Correlation_scatterplots\abeta_"+name)
        plt.close()
    #for j in range(len(abeta)):
    #    print(abeta[j], params[j,i],)
    print(f"{name:25s}  rho={rho:+.3f}  p={p:.3e}")

plt.scatter(abeta, tau)
#plt.plot(373+0.82*np.linspace(np.min(tau), np.max(tau), len(tau)), np.linspace(np.min(tau), np.max(tau), len(tau)), label='alternative condition')
plt.plot(813*np.ones(len(tau)), np.linspace(np.min(tau), np.max(tau), len(tau)), color = 'r', label='abeta cutoff')
plt.plot(abeta_stage_1*np.ones(len(tau)), np.linspace(np.min(tau), np.max(tau), len(tau)), color = 'r')#, label='abeta 1')
plt.plot(abeta_stage_2*np.ones(len(tau)), np.linspace(np.min(tau), np.max(tau), len(tau)), color = 'r')#, label='abeta 2')
plt.plot(abeta_stage_3*np.ones(len(tau)), np.linspace(np.min(tau), np.max(tau), len(tau)), color = 'r')#, label='abeta 3')
plt.plot(np.linspace(np.min(abeta), np.max(abeta), len(abeta)), 53*np.ones(len(abeta)), color = 'g', label='tau cutoff')
plt.plot(np.linspace(np.min(abeta), np.max(abeta), len(abeta)), tau_stage_1*np.ones(len(abeta)), color = 'g')#, label='tau 1')
plt.plot(np.linspace(np.min(abeta), np.max(abeta), len(abeta)), tau_stage_2*np.ones(len(abeta)), color = 'g')#, label='tau 2')
plt.plot(np.linspace(np.min(abeta), np.max(abeta), len(abeta)), tau_stage_3*np.ones(len(abeta)), color = 'g')#, label='tau 3')
plt.title('abeta vs tau')
plt.xlabel('CSF Abeta')
plt.ylabel('CSF Tau')
plt.grid()
plt.legend()
plt.savefig(r"C:\Users\User\OneDrive - University of Pisa\Desktop\abeta_vs_tau")
plt.close()

plt.scatter(abeta, inferred_patients_stages[valid, 1])
plt.xlabel('CSF Abeta')
plt.ylabel('Inferred stage')
plt.grid()
plt.savefig(r"C:\Users\User\OneDrive - University of Pisa\Desktop\abeta_vs_inferre_stage")
plt.close()

plt.scatter(tau, inferred_patients_stages[valid, 0])
plt.xlabel('CSF Tau')
plt.ylabel('Inferred stage')
plt.grid()
plt.savefig(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tau_vs_inferre_stage")
plt.close()

########## ovviamente questi sono 3 stage mentre noi ne abbiamo 5, per come abbiamo strutturato i nostri stages
########## la corrispondenda stages_reali vs stages_simulati sarà la seguente:
########## stage_reale 1 --> stage_simulato 1
########## stage_reale 2 --> stage_simulato 2
########## stage_reale 3 --> stage_simulato 3,4,5
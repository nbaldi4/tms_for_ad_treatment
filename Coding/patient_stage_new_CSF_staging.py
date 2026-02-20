import pandas as pd
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import global_variable_creation as gv
from sklearn.feature_selection import mutual_info_regression

from data_analysis_utils import real_vs_stages_PSD, real_vs_stages_FC, to_common_indices, macro_fc, analyze_regular_data, analyze_reduced_data



import numpy as np
import matplotlib.pyplot as plt


############################################### DATI RELATIVI A CSF ##########################################################
abeta_max_CSF = 241.87
abeta_min_CSF = 130.53
abeta_delta_CSF = abeta_min_CSF-abeta_max_CSF

years_abeta_HC_to_preclinical_CSF = 32.6
years_abeta_preclinical_to_MCI_CSF = 5.2
years_abeta_MCI_to_dementia_CSF = 7.4
years_abeta_tot_CSF = years_abeta_HC_to_preclinical_CSF + years_abeta_preclinical_to_MCI_CSF + years_abeta_MCI_to_dementia_CSF


tau_max_CSF = 47.07
tau_min_CSF = 25.10
tau_delta_CSF = tau_max_CSF-tau_min_CSF
years_tau_HC_to_MCI_CSF = 23.4
years_tau_MCI_to_dementia_CSF = 5.6
years_tau_tot_CSF = years_tau_HC_to_MCI_CSF + years_tau_MCI_to_dementia_CSF

def sigmoid(x, slope=1, center=0):
    """
    Equazione della sigmoide parametrica.
    slope: controlla la pendenza (r).
    center: controlla lo spostamento orizzontale (v0).
    """
    return 1 / (1 + np.exp(-slope * (x - center)))

def inverse_sigmoid(y, slope=1, center=0):
    """
    Equazione della sigmoide parametrica.
    slope: controlla la pendenza (r).
    center: controlla lo spostamento orizzontale (v0).
    """
    return center + (1 / slope) * np.log(y / (1 - y))

# 1. Creiamo un intervallo di valori per l'asse X (es. da -10 a 10)
x = np.linspace(-75, 50, 10000)

# 2. Calcoliamo i valori della sigmoide
y_abeta_CSF = sigmoid(x, slope=-2*np.log(19)/years_abeta_tot_CSF, center=-years_abeta_tot_CSF/2)
y_tau_CSF = sigmoid(x, slope=2*np.log(19)/years_tau_tot_CSF, center=-years_tau_tot_CSF/2)
#sappiamo che il cutoff è 192, quindi normalizzandolo sarà (192-130.53)/(241.87-130.53)
cutoff = (192-130.53)/(241.87-130.53)
############################################### DATI RELATIVI A IMAGING ##########################################################
years_abeta_HC_to_MCI_imaging = 22.6 # from 0.74 to 0.95
years_abeta_MCI_to_dementia_imaging = 9.1 # from 0.95 to 1.02
years_abeta_tot_imaging = years_abeta_HC_to_MCI_imaging + years_abeta_MCI_to_dementia_imaging

y_abeta_imaging = sigmoid(x, slope=-2*np.log(19)/years_abeta_tot_imaging, center=-years_abeta_tot_imaging/2)

############# Per valutare l'imaging di tau suppongo che l'accumulo di abeta visibile nell'imaging rispetto alla diminuzione di abeta nel CSF, 
############# in proporzione sia uguale all'accumulo di tau visibile nell'imaging rispetto all'aumento di tau nel CSF

############# la scala temporale di abeta CSF è years_abeta_tot_CSF=45.2 mentre la scala temporale di abeta imaging è years_abeta_tot_imaging=31.7
############# tennedo presente questo facciamo la proporzione con le scale temporali di tau
############# la scala temporale di tau   CSF è years_tau_tot_CSF=29.0   quindi la scala temporale di tau imaging sarà years_tau_tot_imaging=31.7*29.0/45.2=20.339;
############# che approssimeremo con 20.3 per coerenza con le cifre significative delle altre scale temporali
############# l'unica "scocciatura" è che ci stiamo basando sull'unico valore che non è misurato ma inferito, però mi sembra comunque una buona guess

years_tau_tot_imaging = 20.3

y_tau_imaging = sigmoid(x, slope=2*np.log(19)/years_tau_tot_imaging, center=-years_tau_tot_imaging/2)

############# infine decidiamo come determinare gli stage: nell'articolo non vengono specificati stage se non HC, preclinical, MCI e dementia,
############# ma questi non sono stage basati sui valori delle proteine visibili nell'imaging, che invece è ciò su cui ci basiamo nelle simulazioni
############# la scelta che faccio quindi è arbitraria ma, avendo delle curve in funzione del tempo sia per le proteine visibili nell'imaging sia 
############# per le proteine visibili nel CSF le quantità saranno autoconsistenti
############# scelgo dunque, anche basandomi sul fatto che abbiamo considerato sano una variazione di tau entro il 5% e un primo stage di tau
############# nel momento in cui siamo al 28%, sceglo di tenere i 3 stage nel momento in cui tau imaging sarà al 28%, 78% e 95%
############# in modo da conservare la buona evoluzione che avevamo ottenuto per tau e in modo da campionare la curva più ripida che abbiamo
############# così da evitare cambiamenti troppo repetini


#print('CSF abeta level at stage I: ',sigmoid(x=-years_abeta_tot_CSF, slope=-2*np.log(19)/years_abeta_tot_CSF, center=-years_abeta_tot_CSF/2))
#print('CSF abeta level at stage II: ',sigmoid(x=-years_abeta_MCI_to_dementia_CSF - years_abeta_preclinical_to_MCI_CSF, slope=-2*np.log(19)/years_abeta_tot_CSF, center=-years_abeta_tot_CSF/2))
#print('CSF abeta level at stage III: ',sigmoid(x=-years_abeta_MCI_to_dementia_CSF, slope=-2*np.log(19)/years_abeta_tot_CSF, center=-years_abeta_tot_CSF/2))
#print('CSF tau level at stage I: ',sigmoid(x=-27.6, slope=2*np.log(19)/years_tau_tot_CSF, center=-years_tau_tot_CSF/2))
#print('CSF tau level at stage II: ',sigmoid(x=-12.6, slope=2*np.log(19)/years_tau_tot_CSF, center=-years_tau_tot_CSF/2))
#print('CSF tau level at stage III: ',sigmoid(x=-7.4, slope=2*np.log(19)/years_tau_tot_CSF, center=-years_tau_tot_CSF/2))

print('CSF abeta preclinical (cutoff) level: 192 (labeled in the article)')
print('CSF abeta preclinical (cutoff) percentage: (241.87-192)/241.87=',(241.87-192)/241.87)
print('CSF tau preclinical (cutoff) level (inferred): 25.10 + 25.10*(241.87-192)/241=',25.10 + 25.10*(241.87-192)/241)
#print('CSF abeta level at stage I: ',sigmoid(x=years_stage_1, slope=-2*np.log(19)/years_abeta_tot_CSF, center=-years_abeta_tot_CSF/2))
print('CSF abeta level at stage I: ',inverse_sigmoid(y=cutoff, slope=-2*np.log(19)/years_abeta_tot_CSF, center=-years_abeta_tot_CSF/2))
#print('CSF abeta level at stage II: ',sigmoid(x=years_stage_2, slope=-2*np.log(19)/years_abeta_tot_CSF, center=-years_abeta_tot_CSF/2))
#print('CSF abeta level at stage III: ',sigmoid(x=years_stage_3, slope=-2*np.log(19)/years_abeta_tot_CSF, center=-years_abeta_tot_CSF/2))
print('CSF tau level at stage I: ',inverse_sigmoid(y=cutoff, slope=2*np.log(19)/years_tau_tot_CSF, center=-years_tau_tot_CSF/2))
#print('CSF tau level at stage II: ',sigmoid(x=years_stage_2, slope=2*np.log(19)/years_tau_tot_CSF, center=-years_tau_tot_CSF/2))
#print('CSF tau level at stage III: ',sigmoid(x=years_stage_3, slope=2*np.log(19)/years_tau_tot_CSF, center=-years_tau_tot_CSF/2))
#print('imaging abeta level at stage I: ',sigmoid(x=years_stage_1, slope=2*np.log(19)/years_abeta_tot_imaging, center=-years_abeta_tot_imaging/2))
#print('imagin abeta level at stage II: ',sigmoid(x=years_stage_2, slope=2*np.log(19)/years_abeta_tot_imaging, center=-years_abeta_tot_imaging/2))
#print('imagin abeta level at stage III: ',sigmoid(x=years_stage_3, slope=2*np.log(19)/years_abeta_tot_imaging, center=-years_abeta_tot_imaging/2))
#print('imagin tau level at stage I: ',sigmoid(x=years_stage_1, slope=2*np.log(19)/years_tau_tot_imaging, center=-years_tau_tot_imaging/2))
#print('imagin tau level at stage II: ',sigmoid(x=years_stage_2, slope=2*np.log(19)/years_tau_tot_imaging, center=-years_tau_tot_imaging/2))
#print('imagin tau level at stage III: ',sigmoid(x=years_stage_3, slope=2*np.log(19)/years_tau_tot_imaging, center=-years_tau_tot_imaging/2))
# 3. Disegniamo il grafico
plt.figure(figsize=(10, 6))

plt.plot(x, y_abeta_CSF, label='Abeta CSF', linewidth=2)
plt.plot(inverse_sigmoid(y=cutoff, slope=-2*np.log(19)/years_abeta_tot_CSF, center=-years_abeta_tot_CSF/2)*np.ones(10), np.linspace(0,1,10), label='stage 1-5 abeta')
plt.plot((inverse_sigmoid(y=cutoff, slope=-2*np.log(19)/years_abeta_tot_CSF, center=-years_abeta_tot_CSF/2)/2)*np.ones(10), np.linspace(0,1,10), label='stage 2-4 abeta')
plt.plot(np.zeros(10), np.linspace(0,1,10), label='stage 3 abeta')
plt.plot(np.linspace(-75,50,10), np.ones(10)*sigmoid(x=inverse_sigmoid(y=cutoff, slope=-2*np.log(19)/years_abeta_tot_CSF, center=-years_abeta_tot_CSF/2)/2, slope=-2*np.log(19)/years_abeta_tot_CSF, center=-years_abeta_tot_CSF/2), label='abeta level 2-4')
plt.plot(np.linspace(-75,50,10), np.ones(10)*sigmoid(x=inverse_sigmoid(y=cutoff, slope=-2*np.log(19)/years_abeta_tot_CSF, center=-years_abeta_tot_CSF/2), slope=-2*np.log(19)/years_abeta_tot_CSF, center=-years_abeta_tot_CSF/2), label='abeta level 1-5')
plt.plot(np.linspace(-80,40,10), np.ones(10)*0.05, label='abeta level 3')

# Aggiungiamo dettagli al grafico
plt.axhline(0.5, color='gray', linewidth=0.5, linestyle='--') # Linea di metà altezza
plt.axvline(0, color='gray', linewidth=0.5)                  # Asse Y
plt.title("Protein level", fontsize=14)
plt.xlabel("Years respect to AD dementia", fontsize=12)
plt.ylabel("Normalized CSF Abeta level", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

#plt.show()

print('valore sigmoide: ', )

plt.figure(figsize=(10, 6))

plt.plot(x, y_tau_CSF, label='Tau CSF', linewidth=2)
plt.plot((inverse_sigmoid(y=cutoff, slope=2*np.log(19)/years_tau_tot_CSF, center=-years_tau_tot_CSF/2)*2)*np.ones(10), np.linspace(0,1,10), label='stage 1 tau')
plt.plot((inverse_sigmoid(y=cutoff, slope=2*np.log(19)/years_tau_tot_CSF, center=-years_tau_tot_CSF/2)*3/2)*np.ones(10), np.linspace(0,1,10), label='stage 2 tau')
plt.plot(inverse_sigmoid(y=cutoff, slope=2*np.log(19)/years_tau_tot_CSF, center=-years_tau_tot_CSF/2)*np.ones(10), np.linspace(0,1,10), label='stage 3 tau')
plt.plot((inverse_sigmoid(y=cutoff, slope=2*np.log(19)/years_tau_tot_CSF, center=-years_tau_tot_CSF/2)/2)*np.ones(10), np.linspace(0,1,10), label='stage 4 tau')
plt.plot(np.zeros(10), np.linspace(0,1,10), label='stage 5 tau')
plt.plot(np.linspace(-75,50,10), np.ones(10)*sigmoid(x=inverse_sigmoid(y=cutoff, slope=2*np.log(19)/years_tau_tot_CSF, center=-years_tau_tot_CSF/2)*2, slope=2*np.log(19)/years_tau_tot_CSF, center=-years_tau_tot_CSF/2), label='tau level 1')
plt.plot(np.linspace(-75,50,10), np.ones(10)*sigmoid(x=inverse_sigmoid(y=cutoff, slope=2*np.log(19)/years_tau_tot_CSF, center=-years_tau_tot_CSF/2)*3/2, slope=2*np.log(19)/years_tau_tot_CSF, center=-years_tau_tot_CSF/2), label='tau level 2')
plt.plot(np.linspace(-75,50,10), np.ones(10)*sigmoid(x=inverse_sigmoid(y=cutoff, slope=2*np.log(19)/years_tau_tot_CSF, center=-years_tau_tot_CSF/2), slope=2*np.log(19)/years_tau_tot_CSF, center=-years_tau_tot_CSF/2), label='tau level 3')
plt.plot(np.linspace(-75,50,10), np.ones(10)*sigmoid(x=inverse_sigmoid(y=cutoff, slope=2*np.log(19)/years_tau_tot_CSF, center=-years_tau_tot_CSF/2)/2, slope=2*np.log(19)/years_tau_tot_CSF, center=-years_tau_tot_CSF/2), label='tau level 4')
plt.plot(np.linspace(-75,50,10), np.ones(10)*cutoff, label='cutoff')
plt.plot(np.linspace(-75,50,10), np.ones(10)*0.95, label='tau level 5')
plt.plot(np.linspace(-75,50,10), np.ones(10)*0.05, label='lowest value')

# Aggiungiamo dettagli al grafico
plt.axhline(0.5, color='gray', linewidth=0.5, linestyle='--') # Linea di metà altezza
plt.axvline(0, color='gray', linewidth=0.5)                  # Asse Y
plt.title("Protein level", fontsize=14)
plt.xlabel("Years respect to AD dementia", fontsize=12)
plt.ylabel("Normalized CSF Tau level", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

#plt.show()





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

#In questo nuovo articolo invece si suppone che la sinusoide sia tale che il cutoff nega raggiunto nello stadio 1 per abeta e nello stadio 3 per tau
#Dato che riconoscere il livello di abeta come non motono ci ha dato una buona ripartizione degli stage, continueremo a supporre che il livello di abeta CSF prima cresca e poi decresca

abeta_delta = 1024.17 - 552.71
abeta_stage_1 = 552.71 + abeta_delta*sigmoid(x=inverse_sigmoid(y=cutoff, slope=-2*np.log(19)/years_abeta_tot_CSF, center=-years_abeta_tot_CSF/2), slope=-2*np.log(19)/years_abeta_tot_CSF, center=-years_abeta_tot_CSF/2)
abeta_stage_2 = 552.71 + abeta_delta*sigmoid(x=inverse_sigmoid(y=cutoff, slope=-2*np.log(19)/years_abeta_tot_CSF, center=-years_abeta_tot_CSF/2)/2, slope=-2*np.log(19)/years_abeta_tot_CSF, center=-years_abeta_tot_CSF/2)
abeta_stage_3 = 552.71 + abeta_delta*0.05
print('abeta_stage_1', abeta_stage_1)
print('abeta_stage_2', abeta_stage_2)
print('abeta_stage_3', abeta_stage_3)


tau_delta = 80.81 - 28.81
tau_stage_1 = 28.81 + tau_delta*sigmoid(x=inverse_sigmoid(y=cutoff, slope=2*np.log(19)/years_tau_tot_CSF, center=-years_tau_tot_CSF/2)*2, slope=2*np.log(19)/years_tau_tot_CSF, center=-years_tau_tot_CSF/2)
tau_stage_2 = 28.81 + tau_delta*sigmoid(x=inverse_sigmoid(y=cutoff, slope=2*np.log(19)/years_tau_tot_CSF, center=-years_tau_tot_CSF/2)*3/2, slope=2*np.log(19)/years_tau_tot_CSF, center=-years_tau_tot_CSF/2)
tau_stage_3 = 28.81 + tau_delta*sigmoid(x=inverse_sigmoid(y=cutoff, slope=2*np.log(19)/years_tau_tot_CSF, center=-years_tau_tot_CSF/2), slope=2*np.log(19)/years_tau_tot_CSF, center=-years_tau_tot_CSF/2)
tau_stage_4 = 28.81 + tau_delta*sigmoid(x=inverse_sigmoid(y=cutoff, slope=2*np.log(19)/years_tau_tot_CSF, center=-years_tau_tot_CSF/2)/2, slope=2*np.log(19)/years_tau_tot_CSF, center=-years_tau_tot_CSF/2)
tau_stage_5 = 28.81 + tau_delta*0.95
print('tau_stage_1', tau_stage_1)
print('tau_stage_2', tau_stage_2)
print('tau_stage_3', tau_stage_3)
print('tau_stage_4', tau_stage_4)
print('tau_stage_5', tau_stage_5)


#########con questa divisione in stage la performance dell'algoritmo sale a 10 match e 4 mismatch di 1 (su 30 confronti totali)
for i in range(len(patient_fc_metrics[:,0])):
    if (prot_lev[i,0] == 0) & (prot_lev[i,1] == 0):
        patients_stages[i,0] = np.nan
        patients_stages[i,1] = np.nan

    if (prot_lev[i,1] < tau_stage_1):
        patients_stages[i,1] = 0
        
    if (tau_stage_1 <= prot_lev[i,1] < tau_stage_2):
        patients_stages[i,1] = 1

    if (tau_stage_2 <= prot_lev[i,1] < tau_stage_3):
        patients_stages[i,1] = 2
        
    if (tau_stage_3 <= prot_lev[i,1] < tau_stage_4):
        patients_stages[i,1] = 3
        
    if (tau_stage_4 <= prot_lev[i,1] < tau_stage_5):
        patients_stages[i,1] = 4
        
    if ( prot_lev[i,1] >= tau_stage_5):
        patients_stages[i,1] = 5

    if (prot_lev[i,0] < abeta_stage_1) & (prot_lev[i,1] < tau_stage_1):
        patients_stages[i,0] = 0

    if (abeta_stage_1 <= prot_lev[i,0] < abeta_stage_2) & (prot_lev[i,1] < tau_stage_1):
        patients_stages[i,0] = 1

    if (prot_lev[i,0] < abeta_stage_2) & (prot_lev[i,1] >= tau_stage_1):
        patients_stages[i,0] = 5

    if (abeta_stage_2 <= prot_lev[i,0] < abeta_stage_3) & (prot_lev[i,1] < tau_stage_1):
        patients_stages[i,0] = 2

    if (abeta_stage_2 <= prot_lev[i,0] < abeta_stage_3) & (prot_lev[i,1] >= tau_stage_1):
        patients_stages[i,0] = 4

    if (prot_lev[i,0] <= abeta_stage_3):
        patients_stages[i,0] = 3

    idx_flat = np.argmin(distances[i, :, :])

    # 2. Convertilo in coordinate (riga, colonna)
    #riga, colonna = np.unravel_index(idx_flat, distances[i, :, :].shape)
    riga = np.argmin(distances_abeta[i, :])
    colonna = np.argmin(distances_tau[i, :])
    if prot_lev[i,0] != 0:
        print(i, ': ', prot_lev[i], '; ', patients_stages[i], '; ', riga, ' ', colonna)

########## ovviamente questi sono 3 stage mentre noi ne abbiamo 5, per come abbiamo strutturato i nostri stages
########## la corrispondenda stages_reali vs stages_simulati sarà la seguente:
########## stage_reale 1 --> stage_simulato 1
########## stage_reale 2 --> stage_simulato 2
########## stage_reale 3 --> stage_simulato 3,4,5
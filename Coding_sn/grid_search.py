import numpy as np
import pandas as pd
from joblib import Parallel, delayed
# Altri import
import time
import os
import matplotlib.pyplot as plt
import sys 

# tue funzioni (Assumiamo che siano accessibili)
from utils import preprocess, simulate
from data_analysis_utils import real_vs_simulated_FC_single_value, real_vs_simulated_PSD_single_value
import global_variable_creation as gv

# --- 1. DATI E CARICAMENTO ---

# Array di dati clinici (mantenuti)
# ... [Abeta_patients e tau_patients] ...

# Caricamento dati (Assumiamo che i percorsi siano corretti e i dati caricati)
MCI_FC_data_path = r'C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\MCI_data\fc_mci_preview.npy'
MCI_PSD_data_path = r'C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\MCI_data\psd_mci_preview.npy'

try:
    mci_fc = np.load(MCI_FC_data_path) 
    mci_psd = np.load(MCI_PSD_data_path)
    print("Dati MCI_FC e MCI_PSD caricati con successo.")
except FileNotFoundError as e:
    print(f"ERRORE: Impossibile trovare il file: {e.filename}")
    sys.exit(1)


# --- 2. DEFINIZIONE DEL PROBLEMA E PARAMETRI ---

# Parametri G-sweep
samples = 50
g_range = np.logspace(-5, 5, samples)

# Indici pazienti fissi (usati per l'analisi single-value)
tau_index = 11
abeta_index = 23
ad_index = 13

# Parametri Joblib
N_JOBS = -1 # Usa tutti i core disponibili

def evaluate_simulation(g, sim_time=20000):
    """
    Funzione di valutazione per un singolo valore di G.
    Ritorna G e le metriche.
    """
    # I dati mci_fc e mci_psd sono accessibili qui come variabili globali.

    # Esegue la simulazione
    ttavg, tavg, teeg, eeg = simulate(sim_time=sim_time, g=g)
    teeg, eeg = preprocess(teeg, eeg)

    # Confronto FC
    regular_mae, reduced_mae = real_vs_simulated_FC_single_value(
        fc_healty=mci_fc, 
        index=tau_index,
        time_series=eeg,
        return_value=True
    )

    # Confronto PSD
    psd_mae, peak_freq = real_vs_simulated_PSD_single_value(
        matrix=mci_psd, 
        index=tau_index,
        time_series=eeg,
        return_value=True
    )
    
    # Ritorna G + tutte e 4 le metriche
    return g, regular_mae, reduced_mae, psd_mae, peak_freq

# --- 3. ESECUZIONE IN PARALLELO CON JOBLIB ---

print(f"\n🚀 Avvio simulazione in parallelo per {samples} valori di G su {os.cpu_count()} core...")

# L'esecuzione parallela sostituisce il tuo ciclo 'for g in g_range:'
# delayed(evaluate_simulation)(g=g) crea una chiamata funzione per ogni g
# Parallel(n_jobs=N_JOBS)(...) esegue tutte le chiamate in parallelo
all_results = Parallel(n_jobs=10)(
    delayed(evaluate_simulation)(g=g) for g in g_range
)

# --- 4. ELABORAZIONE E SALVATAGGIO DEI RISULTATI ---

# Definizione delle colonne
columns = ["G", "regular_mae", "reduced_mae", "psd_mae", "peak_freq"]

# Creazione del DataFrame Pandas
results_df = pd.DataFrame(all_results, columns=columns)

# Nome del file di output
output_filename = "g_sweep_results.csv"

# Salvataggio su CSV
results_df.to_csv(output_filename, index=False)

print("\n✅ Esecuzione completata.")
print(f"Risultati per lo sweep di G salvati in: {output_filename}")
print(f"Numero di simulazioni completate: {len(results_df)}")
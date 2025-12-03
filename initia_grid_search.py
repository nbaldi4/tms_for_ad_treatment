import numpy as np
import os
import pandas as pd
from joblib import Parallel, delayed
from itertools import product
import time
import matplotlib.pyplot as plt

# --- TUE FUNZIONI ---
from utils import preprocess, simulate, psd
from data_analysis_utils import real_vs_simulated_FC, real_vs_simulated_PSD
import global_variable_creation as gv


# -----------------------------
# 1. DEFINIZIONE DEI PARAMETRI
# -----------------------------

g_values = np.array([0.10134311, 1])#np.linspace(0.1, 2.0, 15)
velocity_values = np.array([6.37285589, 10])#np.linspace(1, 300, 15)
noise_values = np.array([0.05, 0.10001255])#np.array([0.05, 0.1, 0.5, 1, 1.5])
a_values = np.array([0.1, 0.13404924])#np.linspace(0.05, 0.15, 3)
b_values = np.array([0.05, 0.06989446])#np.linspace(0.025, 0.075, 3)

# -----------------------------
# 2. FUNZIONE DI EVALUATION
# -----------------------------
def evaluate_simulation(row, sim_time=20000):
    g, velocity, noise, a, b = row

    ttavg, tavg, teeg, eeg = simulate(sim_time=sim_time, g=g, velocity=velocity, noise=noise, a=a, b=b)
    teeg, eeg = preprocess(teeg, eeg)

    regular_mae, reduced_mae = real_vs_simulated_FC(
        fc_healty=np.load(gv.real_FC_data),
        time_series=eeg,
        return_value=True
    )

    psd_mae = real_vs_simulated_PSD(
        matrix=np.load(gv.real_PSD_data),
        time_series=eeg,
        return_value=True
    )

    return regular_mae, reduced_mae, psd_mae


# -----------------------------
# 3. GENERAZIONE DELLA GRIGLIA
# -----------------------------
grid = list(product(g_values, velocity_values, noise_values, a_values, b_values))

print(f"Numero totale di simulazioni: {len(grid)}")


# -----------------------------
# 4. LANCIO PARALLELO
# -----------------------------
start_time = time.time()

results = Parallel(n_jobs=-1, verbose=10)(
    delayed(evaluate_simulation)(row) for row in grid
)

end_time = time.time()
print(f"Grid search completata in {end_time - start_time:.2f} secondi")


# -----------------------------
# 5. CREAZIONE E SALVATAGGIO RESULTS
# -----------------------------
grid_array = np.array(grid)        # shape = (N,5)
results_array = np.array(results)  # shape = (N,3)

df = pd.DataFrame(
    np.hstack([grid_array, results_array]),
    columns=["g", "velocity", "noise", "a", "b", "M1", "M2", "M3"]
)

df.to_csv("grid_search_results.csv", index=False)
print("Risultati salvati in grid_search_results.csv")

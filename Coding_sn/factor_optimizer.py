import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import r2_score, mean_squared_error
import time
import os
import matplotlib.pyplot as plt

# pymoo
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from pymoo.core.termination import Termination

# tue funzioni
from utils import preprocess, simulate
from data_analysis_utils import real_vs_simulated_FC, real_vs_simulated_PSD
import global_variable_creation as gv

# Parametri fissi (esempio con valori medi dei tuoi range)
fixed_velocity = 150.0
fixed_noise = 2.5
fixed_a = 0.100
fixed_b = 0.050
factor_range = [-1,1]

# Range di g da esplorare (Grid Search 1D)
factor_values = np.linspace(factor_range[0], factor_range[1], num=50) # Esplora 50 punti tra 0.1 e 10

def evaluate_factor(factor_val):

    ttavg, tavg, teeg, eeg = simulate(factor=factor_val)
    teeg, eeg = preprocess(teeg, eeg)

    regular_mae, reduced_mae = real_vs_simulated_FC(
        fc_healty=np.load(gv.HC_FC_data),
        time_series=eeg,
        return_value=True
    )

    psd_mae, peak_freq = real_vs_simulated_PSD(
        matrix=np.load(gv.HC_PSD_data),
        time_series=eeg,
        return_value=True
    )
    return regular_mae, reduced_mae, psd_mae, peak_freq

print(f"Inizio Grid Search su g ({len(factor_values)} punti)...")

    # Esecuzione parallela
results = Parallel(n_jobs=-1)(
    delayed(evaluate_factor)(fact) for fact in factor_values
)
    
# Risultati in DataFrame
columns = ["M1_FC_MAE", "M2_Reduced_FC_MAE", "M3_PSD_MAE", "peak_freq"]
df_results = pd.DataFrame(results, columns=columns)
df_results.insert(0, "factor", factor_values) # Aggiungiamo la colonna g all'inizio
   
# Salvataggio
df_results.to_csv("grid_search_factor_results.csv", index=False)
print("Risultati salvati in grid_search_factor_results.csv")

def plot_factor_sensitivity(df):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Asse sinistro per le metriche di errore (MAE)
    ax1.plot(df["factor"], df["M1_FC_MAE"], label="FC MAE", color="blue", linewidth=2)
    ax1.plot(df["factor"], df["M3_PSD_MAE"], label="PSD MAE", color="green", linewidth=2)
    ax1.set_xlabel("a variability factor")
    ax1.set_ylabel("Error (MAE)")
    ax1.legend(loc="upper left")

    # Asse destro per la frequenza di picco
    ax2 = ax1.twinx()
    ax2.plot(df["factor"], df["peak_freq"], label="Peak Frequency", color="red", linestyle="--")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.legend(loc="upper right")

    plt.title("Sensitivity Analysis: Effect of Coupling (factor) on EEG Metrics")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.savefig("factor_sensitivity_analysis.png")
    plt.show()

plot_factor_sensitivity(df_results)
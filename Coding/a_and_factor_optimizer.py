import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import r2_score, mean_squared_error
import time
import os
import seaborn as sns
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
from matplotlib import cm

factor_range = [-0.4,0.4]
a_range = [0.06, 0.085]
factor_values = np.linspace(factor_range[0], factor_range[1], num=21)
a_values = np.linspace(a_range[0], a_range[1], num=21) 

# 2. Modifica la funzione evaluate per scompattare la tupla
def evaluate(params):
    factor_val, a_val = params  # Unpacking della tupla
    
    # Chiamata alla simulazione (aggiungi i parametri fissi se necessario)
    ttavg, tavg, teeg, eeg = simulate(a=a_val, factor=factor_val)
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

def grid_search(factor_set=factor_values, a_set=a_values):
    # 1. Creazione corretta della griglia di coppie
    couples = []
    for f_val in factor_set:
        for a_val in a_set:
            couples.append((f_val, a_val)) # Crea una lista di tuple (factor, a)

    # 3. Esecuzione parallela
    results = Parallel(n_jobs=-3)(
        delayed(evaluate)(coup) for coup in couples
    )

    # 4. Creazione DataFrame con ENTRAMBI i parametri
    df_results = pd.DataFrame(results, columns=["M1_FC_MAE", "M2_Reduced_FC_MAE", "M3_PSD_MAE", "peak_freq"])

    df_results.insert(0, "a", [c[1] for c in couples])
    df_results.insert(0, "factor", [c[0] for c in couples])

    file_path = "grid_search_2D_results.csv"

    file_exists = os.path.isfile(file_path)

    # Salvataggio in modalità append
    df_results.to_csv(file_path, 
                      mode='a', 
                      index=False, 
                      header=not file_exists)

def plot_3d_surface(file_path=r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Coding\grid_search_2D_results - Copia.csv"):
    values = np.loadtxt(file_path, delimiter=",", skiprows=1)
    values = np.array(values)

    fact = values[:, 0]
    a = values[:, 1]
    m1 = values[:, -4]
    m2 = values[:, -3]
    m3 = values[:, -2]

    m_matrix = np.zeros((int(len(fact)/len(a)), int(len(a)/len(fact)))) #rows --> fact, columns --> a
        

    z_matrix = values[:, 4].reshape(len(fact), len(a)).T

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extremal axis values
    # [x_min, x_max, y_min, y_max]
    extent = [fact.min(), fact.max(), a.min(), a.max()]
    
    im = ax.imshow(z_matrix, cmap='bwr', origin='lower', 
                   extent=extent, aspect='auto') 
    
    ax.set_xlabel('Factor')
    ax.set_ylabel('a value')
    ax.set_title("Grid Search Analysis", fontsize=15)
    
    fig.colorbar(im, ax=ax, label='Error')
    plt.show()

def plot_heatmap_robust(file_path="grid_search_2D_results.csv", metric="M3_PSD_MAE"):
    # 1. Carica i dati con Pandas
    df = pd.read_csv(file_path)

    # 2. Rimuovi i duplicati (opzionale, ma consigliato)
    # Teniamo l'ultima occorrenza di ogni coppia (factor, a)
    df = df.drop_duplicates(subset=['factor', 'a'], keep='last')

    # 3. Crea la Pivot Table
    # Questa operazione riordina 'a' sulle righe e 'factor' sulle colonne
    pivot_df = df.pivot(index='a', columns='factor', values=metric)

    # 4. Estrai i valori per il plot
    z_matrix = pivot_df.values
    x_labels = pivot_df.columns.values
    y_labels = pivot_df.index.values

    # 5. Grafico
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Usiamo extent basandoci sui valori reali trovati nel file
    extent = [x_labels.min(), x_labels.max(), y_labels.min(), y_labels.max()]
    
    im = ax.imshow(z_matrix, 
                   cmap='bwr', 
                   origin='lower', 
                   extent=extent, 
                   aspect='auto')

    # Aggiungiamo i titoli
    ax.set_xlabel('Factor (a variability)')
    ax.set_ylabel('a (mean value)')
    ax.set_title(f"Heatmap di {metric}\n(Dati ordinati e ripuliti)", fontsize=14)

    # Colorbar
    plt.colorbar(im, ax=ax, label='Errore MAE')
    
    plt.show()

    df = pd.read_csv("grid_search_2D_results.csv")
    # Creiamo la pivot
    pivot_df = df.pivot(index='a', columns='factor', values='M3_PSD_MAE')

    plt.figure(figsize=(12, 8))
    # Usiamo sns.heatmap perché gestisce meglio le etichette degli assi con i buchi
    sns.heatmap(pivot_df, cmap='bwr', mask=pivot_df.isnull(), cbar_kws={'label': 'MAE'})
    plt.title("Grid Search con dati mancanti (Buchi bianchi)")
    plt.show()

def heatmap_chat():
    df = pd.read_csv(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Coding\grid_search_2D_results.csv")
    df["factor"] = df["factor"].round(6)
    df["a"] = df["a"].round(6)
    
    heatmap_data = df.pivot_table(
        index="a",
        columns="factor",
        values=[
            "M1_FC_MAE",
            "M2_Reduced_FC_MAE",
            "M3_PSD_MAE"
        ],
        aggfunc="mean"   # oppure np.mean
    )
    z1 = heatmap_data["M1_FC_MAE"].values
    z2 = heatmap_data["M2_Reduced_FC_MAE"].values
    z3 = heatmap_data["M3_PSD_MAE"].values

    z = z1 + z2 + 2 * z3
    masked_data = np.ma.masked_invalid(z)
    # Colormap (copiata per poter modificare i NaN)
    cmap = plt.cm.coolwarm.copy() #plt.cm.viridis.copy() sono altri colori
    cmap.set_bad(color="black")  # celle mancanti nere

    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        masked_data,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        #interpolation='bilinear' #da togliere se non vogliamo che interpoli i dati mancanti
    )

    plt.colorbar(im, label="MAE_combination")

    factor_labels = heatmap_data["M1_FC_MAE"].columns
    plt.xticks(
        ticks=np.arange(len(factor_labels)),
        labels=factor_labels,
        size=5
    )

    plt.yticks(
        ticks=np.arange(len(heatmap_data.index)),
        labels=heatmap_data.index
    )

    plt.xlabel("factor")
    plt.ylabel("a")
    plt.title("MAE distribution")

    plt.tight_layout()
    plt.show()
#grid_search(factor_set=np.linspace(0.02, 0.38, 10), a_set=np.linspace(0.06, 0.07875, 16))
#grid_search(factor_set=np.linspace(-0.38, -0.22, 5), a_set=np.linspace(0.07125, 0.085, 12))
#grid_search(factor_set=np.linspace(-0.8, -0.62, 10), a_set=np.linspace(0.055, 0.075, 17))
heatmap_chat()
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import os
import matplotlib.pyplot as plt

# tue funzioni
from utils import preprocess, simulate
from data_analysis_utils import real_vs_simulated_FC, real_vs_simulated_PSD
import global_variable_creation as gv


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
    results = Parallel(n_jobs=-2)(
        delayed(evaluate)(coup) for coup in couples
    )

    # 4. Creazione DataFrame con ENTRAMBI i parametri
    df_results = pd.DataFrame(results, columns=["M1_FC_MAE", "M2_Reduced_FC_MAE", "M3_PSD_MAE", "peak_freq"])

    df_results.insert(0, "a", [c[1] for c in couples])
    df_results.insert(0, "factor", [c[0] for c in couples])

    file_path = "grid_search_2D_results_with_b_corrected.csv"

    file_exists = os.path.isfile(file_path)

    # Salvataggio in modalità append
    df_results.to_csv(file_path, 
                      mode='a', 
                      index=False, 
                      header=not file_exists)

def heatmap():
    df = pd.read_csv(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Coding\grid_search_2D_results_with_b_corrected.csv")
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

    mean_z1 = np.nanmean(z1)
    mean_z2 = np.nanmean(z2)
    mean_z3 = np.nanmean(z3)

    dev_z1 = (z1 - mean_z1)/mean_z1
    dev_z2 = (z2 - mean_z2)/mean_z2
    dev_z3 = (z3 - mean_z3)/mean_z3

    z = dev_z1 + dev_z2 + 2*dev_z3
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

    # 1. Trova l'indice del valore minimo ignorando i NaN
    # np.nanargmin restituisce l'indice come se l'array fosse piatto (1D)
    idx_min_flat = np.nanargmin(z)

    # 2. Converte l'indice piatto in coordinate (riga, colonna) della matrice
    row_idx, col_idx = np.unravel_index(idx_min_flat, z.shape)

    # 3. Recupera i valori reali di 'a' (indice) e 'factor' (colonne)
    # heatmap_data.index contiene i valori di 'a'
    # heatmap_data["M1_FC_MAE"].columns contiene i valori di 'factor'
    a_opt = heatmap_data.index[row_idx]
    factor_opt = heatmap_data["M1_FC_MAE"].columns[col_idx]
    z_min_val = z[row_idx, col_idx]

    print(f"\n--- Risultati Ottimizzazione ---")
    print(f"Valore minimo di z: {z_min_val:.4f}")
    print(f"Parametri ottimali trovati:")
    print(f"  > factor: {factor_opt}")
    print(f"  > a:      {a_opt}")
#grid_search(factor_set=np.linspace(0.02, 0.38, 10), a_set=np.linspace(0.06, 0.07875, 16))
#grid_search(factor_set=np.linspace(-0.38, -0.22, 5), a_set=np.linspace(0.07125, 0.085, 12))
#grid_search(factor_set=np.linspace(-0.9, -0.62, 15), a_set=np.linspace(0.055, 0.075, 17))
##grid_search(factor_set=np.linspace(-0.92, -0.04, 23), a_set=np.linspace(0.055, 0.085, 25))
##grid_search(factor_set=np.linspace(0.00, 0.40, 11), a_set=np.linspace(0.06, 0.105, 37))
heatmap()
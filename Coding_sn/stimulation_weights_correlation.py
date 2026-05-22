import os

import numpy as np
from tvb.simulator.lab import *
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import stats
import matplotlib.patches as mpatches
from scipy.stats import linregress

import pandas as pd

def corr_search(
    abeta = 1,
    tau = 1,
    spot = 0,
    freq = 10,
    intensity = 0.5):

    file_name = f"super_th_reg_a{abeta}_t{tau}_f{freq}_s{spot}"
    base_dir = os.path.dirname(os.path.abspath(__file__))  # cartella di utils.py
    path = os.path.join(base_dir, '..', 'Tms_results_end', file_name)
    df = pd.read_pickle(path)

    

    intensityplus = intensity + 0.01
    intensityminus = intensity - 0.01
    df=df.query("sf <= @intensityplus and sf >= @intensityminus")
        
    pre_stim_amp = df["mean_amp_pre"].reset_index(drop=True)
    post_stim_amp = df["mean_amp_post"].reset_index(drop=True)
    baseline = df["mean_baseline"].reset_index(drop=True)
    threshold = df["is_greater"].reset_index(drop=True)
    
    norm = 0.2
    second_order_weights = (np.abs(post_stim_amp-pre_stim_amp))/norm
    second_order_weights = second_order_weights/np.max(second_order_weights)

    first_order_weights = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\Pesi_protocolli_TMS\PCU_1_stimulus_vector_TVB.npy")
    first_order_weights = first_order_weights/np.max(first_order_weights)

    mask = first_order_weights > 0
    x = first_order_weights[mask]
    y = second_order_weights[mask]

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    line = slope * x + intercept

# 1. Crea la figura e assegnala alla variabile 'fig'
    fig = plt.figure(figsize=(12, 10)) 
    
    # 2. Crea l'asse 2D (rimuovi projection='3d')
    ax = fig.add_subplot(111) 
    
    # 3. Usa 'ax' per plottare, è più sicuro di 'plt'
    ax.scatter(x, y, alpha=0.6, c='royalblue', edgecolors='k', s=80, label='Brain Regions')
    
    # 4. Retta di regressione
    ax.plot(x, line, color='red', linewidth=3, 
            label=f'Fit lineare (R²={r_value**2:.3f})')
    
    # 5. Tick params (attenzione: 70 è enorme per un grafico 2D, prova 20 o 30)
    ax.tick_params(axis='both', labelsize=20, pad=10)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    # 6. Label e Titolo
    ax.set_xlabel('TMS E-Field Intensity', fontsize=22)
    ax.set_ylabel('Parameter Variation', fontsize=22)
    ax.set_title(f'R² = {r_value**2:.3f}', fontsize=25)
    
    #ax.legend(fontsize=15)
    ax.grid(True, linestyle='--', alpha=0.7)

    title = f"corr_a{abeta}_t{tau}_s{spot}_f{freq}_i{intensity}.png"

    plt.tight_layout()
    plt.savefig(os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop\Thesis\Immagini_tesi\Results\Stages_grid_correlations", title))
    plt.close()
    

def main():
    couples = [("0","1"), ("0","3"), ("1","0"), ("3","0"), ("5","0"), ("5","3"), ("1","1"), ("3","3"), ("5","5")]
    for cop in couples:
        for int in [0.5, 1, 5, 10, 15, 20]:
            for sp in [0,1,2,3,4,5,6,7,8,9]:
                corr_search(abeta=cop[0], tau=cop[1], intensity=int, spot=sp)
    
if __name__ == "__main__":
    main()
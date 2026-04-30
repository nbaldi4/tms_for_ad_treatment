import os

import matplotlib.pyplot as plt
import pandas as pd

all_dist = pd.read_pickle(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Compare_cure_n02\closest_stage_all_pairs_Fa_Fbe_Pdobr_Paobr_norm.pkl")
folder_path = r"C:\Users\User\OneDrive - University of Pisa\Desktop\New_stages_heatmap_n02"
os.makedirs(folder_path, exist_ok=True)
freq = ['high']#, 'low']
intensities = [1,10,100,500]
b_stim_values = [0,1,3,5]
t_stim_values = [0,1,3,5]
for b_stim in b_stim_values:
    for t_stim in t_stim_values:
        for sp in range(4):
            for f in freq:
                for inte in intensities:
                    print('hey')
                    b3t0sp0fh = all_dist.query("b_stim == @b_stim and t_stim == @t_stim and sp == @sp and f == @f and inte == @inte")
                    
                    if b3t0sp0fh.empty:
                        continue 

                    heatmap_data = b3t0sp0fh.pivot(index='b_ns', columns='t_ns', values='distance')

                    fig = plt.figure(figsize=(8, 8)) 
                    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
                    
                    # viridis_r rende il minimo (best match) più chiaro e visibile
                    im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto', origin='lower') 
                    
                    # Titolo per il grafico (con LaTeX)
                    title_text = fr'New stage: A$\beta_{{s}}$={b_stim}, $\tau_{{s}}$={t_stim}, $sp$={sp}, $f$={f}, $int$={inte}'
                    ax.set_title(title_text, fontsize=15)
                    ax.set_xlabel(r'$\tau_{ns}$ stage', fontsize=12)
                    ax.set_ylabel(r'A$\beta_{ns}$ stage', fontsize=12)
                    
                    cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
                    fig.colorbar(im, cax=cax, label='Normalized distance')

                    # 2. Creazione nome file "pulito" per Windows
                    filename = f"heatmap_b{b_stim}_t{t_stim}_sp{sp}_{f}_i{inte}.png"
                    save_path = os.path.join(folder_path, filename)
                    
                    # 3. Salvataggio e chiusura
                    plt.savefig(save_path, dpi=300) # 300 dpi è qualità da stampa per la tesi
                    plt.close(fig) # LIBERA LA RAM: importantissimo per 360 grafici



print(f"Salvataggio completato! Trovi tutto in: {folder_path}")
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


all_dist = pd.read_pickle(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Compare_cure_n02_th18_new_var\closest_stage_all_pairs_Pdobr_Ptobr_Paobr_Pbobr_norm.pkl")
folder_path = r"C:\Users\User\OneDrive - University of Pisa\Desktop\New_stages_heat_n02_th18_new_var\Pdobr_Ptobr_Paobr_Pbobr"#r"C:\Users\User\OneDrive - University of Pisa\Desktop\New_stages_hist_n02_th18_postau"
os.makedirs(folder_path, exist_ok=True)

freq = ['high','low']
intensities = [0.5, 1, 2, 4, 6, 8, 10, 100, 500]
intensities = [0.5, 1, 2, 4, 6, 8, 10, 100, 500]
b_stim_values = [0,1,3,5]
t_stim_values = [0,1,3,5]
sps = [0,1,2,3,4, 5, 6, 7,8,9]
sps = [4,5,6,7]

possible_disease_values = sorted(set(b**2 + t**2 for b in range(6) for t in range(6)))
def plot_hists():
    for b_stim in b_stim_values:
        for t_stim in t_stim_values:
            for sp in sps:
                for f in freq:
                    for inte in intensities:
                        print('b ', b_stim, ' t ', t_stim)
                        subset = all_dist.query("b_stim == @b_stim and t_stim == @t_stim and sp == @sp and f == @f and inte == @inte")

                        if subset.empty:
                            continue

                        # disease_value
                        subset = subset.copy()
                        subset['disease_value'] = subset['b_ns']**2 + subset['t_ns']**2

                        # if two stages have the same disease_value we compute the mean between the distances
                        grouped = subset.groupby('disease_value')['distance'].mean().reset_index()
                        grouped.columns = ['disease_value', 'mean_distance']

                        bar_heights = grouped.set_index('disease_value').reindex(possible_disease_values)['mean_distance'].fillna(np.nan)

                        # color best and start disease stage
                        start_disease_value = b_stim**2 + t_stim**2
                        min_disease_value = possible_disease_values[int(np.nanargmin(bar_heights))]
                        x_pos = np.arange(len(possible_disease_values))

                        colors = []
                        for v in possible_disease_values:
                            if v == start_disease_value:
                                colors.append('crimson')
                            elif v == min_disease_value:
                                colors.append('green')
                            else:
                                colors.append('steelblue')

                        fig, ax = plt.subplots(figsize=(12, 5))

                        ax.bar(x_pos, bar_heights, color=colors, edgecolor='k', linewidth=0.4)
                        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

                        ax.set_xticks(x_pos)
                        ax.set_xticklabels(possible_disease_values, rotation=45, fontsize=8)
                        ax.set_xlim(-1, len(possible_disease_values))
                        ax.set_xlabel(r'Valore della malattia ($A\beta^2 + \tau^2$)', fontsize=12)
                        ax.set_ylabel('Distanza media', fontsize=11)
                        ax.set_ylim(0, 3.5)
                        ax.set_yticks(np.linspace(0, 3.5, 15))
                        ax.grid()
                        ax.set_title(
                            fr'A$\beta_s$={b_stim}, $\tau_s$={t_stim}, sp={sp}, f={f}, int={inte}',
                            fontsize=13
                        )

                        legend_elements = [
                            Patch(facecolor='crimson',   label=f'Stato di partenza (dv={start_disease_value})'),
                            Patch(facecolor='green',     label=f'Distanza minima (dv={min_disease_value})'),
                            Patch(facecolor='steelblue', label='Altri stati'),
                        ]
                        ax.legend(handles=legend_elements, fontsize=10)

                        filename = f"disthist_b{b_stim}_t{t_stim}_i{inte}_sp{sp}_{f}.png"
                        save_path = os.path.join(folder_path, filename)
                        plt.savefig(save_path, dpi=200, bbox_inches='tight')
                        plt.close(fig)

def plot_heatmaps():
    protocols = []
    for f in freq:
        for sp in sps:
            for inte in intensities:
                protocols.append((f, sp, inte))
    
    protocol_labels = [f"{f}, sp{sp}, i{inte}" for (f, sp, inte) in protocols]
    n_protocols = len(protocols)
    n_disease = len(possible_disease_values)

    # --- Primo passaggio: calcola vmax globale ---
    global_max = 0.0

    for b_stim in b_stim_values:
        for t_stim in t_stim_values:
            for p_idx, (f, sp, inte) in enumerate(protocols):
                subset = all_dist.query(
                    "b_stim == @b_stim and t_stim == @t_stim and "
                    "sp == @sp and f == @f and inte == @inte"
                )
                if subset.empty:
                    continue
                subset = subset.copy()
                subset['disease_value'] = subset['b_ns']**2 + subset['t_ns']**2
                local_max = subset['distance'].max()
                if local_max > global_max:
                    global_max = local_max

    print(f"Distanza massima globale: {global_max:.4f}")

    # --- Secondo passaggio: plot con scala comune ---
    for b_stim in b_stim_values:
        for t_stim in t_stim_values:
            start_disease_value = b_stim**2 + t_stim**2
            heat_matrix = np.full((n_protocols, n_disease), np.nan)

            for p_idx, (f, sp, inte) in enumerate(protocols):
                subset = all_dist.query(
                    "b_stim == @b_stim and t_stim == @t_stim and "
                    "sp == @sp and f == @f and inte == @inte"
                )
                if subset.empty:
                    continue

                subset = subset.copy()
                subset['disease_value'] = subset['b_ns']**2 + subset['t_ns']**2
                grouped = subset.groupby('disease_value')['distance'].mean().reset_index()
                grouped.columns = ['disease_value', 'mean_distance']
                bar_heights = (
                    grouped.set_index('disease_value')
                    .reindex(possible_disease_values)['mean_distance']
                    .fillna(np.nan)
                    .values
                )
                heat_matrix[p_idx, :] = bar_heights

            fig, ax = plt.subplots(figsize=(max(10, n_disease * 0.55), max(8, n_protocols * 0.28)))

            im = ax.imshow(
                heat_matrix/global_max,
                aspect='auto',
                cmap='coolwarm',        # blu=basso, rosso=alto
                vmin=0,
                vmax=1,        # scala comune a tutti i grafici
                interpolation='nearest'
            )

            start_col = possible_disease_values.index(start_disease_value)
            ax.axvline(start_col - 0.5, color='black', linewidth=1.5, linestyle='--')
            ax.axvline(start_col + 0.5, color='black', linewidth=1.5, linestyle='--')

            ax.set_xticks(np.arange(n_disease))
            ax.set_xticklabels(possible_disease_values, rotation=45, fontsize=8)
            ax.set_yticks(np.arange(n_protocols))
            ax.set_yticklabels(protocol_labels, fontsize=7)
            ax.set_xlabel(r'Valore della malattia ($A\beta^2 + \tau^2$)', fontsize=12)
            ax.set_ylabel('Protocollo (freq, spot, intensità)', fontsize=11)
            ax.set_title(
                fr'Heatmap distanze — $A\beta_s$={b_stim}, $\tau_s$={t_stim}  '
                fr'(stato partenza dv={start_disease_value})',
                fontsize=13
            )

            cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
            cbar.set_label(f'Distanza media (max={global_max:.2f})', fontsize=10)

            legend_elements = [
                Line2D([0], [0], color='black', linewidth=1.5,
                       linestyle='--', label=f'Stato di partenza (dv={start_disease_value})')
            ]
            ax.legend(handles=legend_elements, fontsize=9, loc='upper right')

            plt.tight_layout()
            filename = f"heatmap_b{b_stim}_t{t_stim}.png"
            save_path = os.path.join(folder_path, filename)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"Salvata: {filename}")

def plot_protocol_ranking():
    protocols = []
    for f in freq:
        for sp in sps:
            for inte in intensities:
                protocols.append((f, sp, inte))

    protocol_labels = [f"{f}, sp{sp}, i{inte}" for (f, sp, inte) in protocols]

    for b_stim in b_stim_values:
        for t_stim in t_stim_values:
            start_disease_value = b_stim**2 + t_stim**2
            results = []  # (label, delta_dv)

            for p_idx, (f, sp, inte) in enumerate(protocols):
                subset = all_dist.query(
                    "b_stim == @b_stim and t_stim == @t_stim and "
                    "sp == @sp and f == @f and inte == @inte"
                )
                if subset.empty:
                    continue

                subset = subset.copy()
                subset['disease_value'] = subset['b_ns']**2 + subset['t_ns']**2

                # Raggruppa per disease value finale e prendi la distanza media
                grouped = subset.groupby('disease_value')['distance'].mean()

                # Stato finale = disease value con distanza minima
                best_dv = grouped.idxmin()
                delta_dv = start_disease_value - best_dv  # positivo = miglioramento

                results.append((protocol_labels[p_idx], delta_dv))

            if not results:
                continue

            # Ordina dal migliore (delta più alto) al peggiore (delta più basso)
            results.sort(key=lambda x: x[1], reverse=True)
            labels, deltas = zip(*results)

            colors = ['steelblue' if d >= 0 else 'tomato' for d in deltas]

            fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.45), 6))

            bars = ax.bar(np.arange(len(labels)), deltas, color=colors, edgecolor='black', linewidth=0.5)

            ax.axhline(0, color='black', linewidth=1.0)

            # Etichetta valore sopra/sotto ogni barra
            for bar, d in zip(bars, deltas):
                va = 'bottom' if d >= 0 else 'top'
                offset = 0.01 * max(abs(v) for v in deltas) * (1 if d >= 0 else -1)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    d + offset,
                    f'{d:.1f}',
                    ha='center', va=va, fontsize=7, rotation=90
                )

            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel(r'$\Delta$ disease value  (initial stage $-$ most similar stage)', fontsize=11)
            ax.set_xlabel('Protocol (freq, spot, intensity)', fontsize=11)
            ax.set_title(
                fr'Ranking protocols — $A\beta_s$={b_stim}, $\tau_s$={t_stim}  '
                fr'(initial disease value={start_disease_value})',
                fontsize=13
            )

            legend_elements = [
                Patch(facecolor='steelblue', edgecolor='black', label='Improvement (↓ dv)'),
                Patch(facecolor='tomato',    edgecolor='black', label='Worsening (↑ dv)'),
            ]
            ax.legend(handles=legend_elements, fontsize=9, loc='upper right')

            plt.tight_layout()
            filename = f"ranking_b{b_stim}_t{t_stim}.png"
            save_path = os.path.join(folder_path, filename)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"Salvata: {filename}")

plot_protocol_ranking()

print("Clutch!")
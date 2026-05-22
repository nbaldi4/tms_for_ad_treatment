import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

def plot_distance_histograms2(df_stim_nostim, df_nostim_nostim, output_dir, log_scale = False):
    """
    Per ogni stadio (b, t) presente in df_stim, genera un istogramma con:
    - asse x: protocolli (sp, f, inte) ordinati dal peggiore al migliore
    - altezza barre: valore di distance (mediato sui rs)
    - colore blu: distance < distance_nostim (migliora)
    - colore rosso: distance >= distance_nostim (peggiora)
    """

    stages = df_stim_nostim[['b', 't']].drop_duplicates().values

    best_protocols = []
    for b, t in stages:
        subset = df_stim_nostim[
            (df_stim_nostim['b'] == b) & (df_stim_nostim['t'] == t)
        ].copy()

        # distanza dello stadio non stimolato corrispondente
        nostim_row = df_nostim_nostim[(df_nostim_nostim['b'] == b) & (df_nostim_nostim['t'] == t)]
        nostim_dist = nostim_row['distance'].values[0] if len(nostim_row) > 0 else np.nan

        # sort from best protocol (minimum distance) to worst protocol (maximum distance)
        subset = subset.sort_values('distance', ascending=True).reset_index(drop=True)

        best_row = subset.iloc[0]  # riga con distanza minima
        best_protocols.append({
            'b':        b,
            't':        t,
            'sp':       best_row['sp'],
            'f':        best_row['f'],
            'inte':     best_row['inte'],
            'distance': best_row['distance']
        })

        labels = [f"sp{int(row.sp)}\n{row.f}\n{row.inte}" for _, row in subset.iterrows()]
        distances = subset['distance'].values
        colors = ['#E24B4A' if d >= nostim_dist else '#378ADD' for d in distances]

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 5))

        bars = ax.bar(range(len(labels)), distances, color=colors, edgecolor='white', linewidth=0.5)

        # dashed lines: non-stimulated distance
        if not np.isnan(nostim_dist):
            ax.axhline(nostim_dist, color='gray', linestyle='--', linewidth=1.2, label=f'no stim (b={b}, t={t}): {nostim_dist:.4f}')
            ax.legend(fontsize=9)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        if log_scale: ax.set_yscale('log')
        ax.set_ylabel('distance', fontsize=10)
        ax.set_title(f'Stimulation stage (abeta,tau)=({b},{t})', fontsize=12)
        ax.set_xlim(-0.5, len(labels) - 0.5)

        blue_patch = mpatches.Patch(color='#378ADD', label='improvement')
        red_patch  = mpatches.Patch(color='#E24B4A', label='worsening')
        ax.legend(handles=[blue_patch, red_patch] + (
            [plt.Line2D([0], [0], color='gray', linestyle='--', label=f'no stim: {nostim_dist:.4f}')]
            if not np.isnan(nostim_dist) else []
        ), fontsize=8, loc='upper right')

        plt.tight_layout()
        if log_scale: output_dir2 = os.path.join(output_dir, 'Semilog')
        else: output_dir2 = os.path.join(output_dir, 'Linear')
        os.makedirs(output_dir2, exist_ok=True)
        fname = f"{output_dir2}/dist_b{b}_t{t}.png"
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"Salvato: {fname}")
    

def plot_distance_histograms(df_stim_nostim, df_nostim_nostim, output_dir, log_scale=False):

    output_dir2 = os.path.join(output_dir, 'Semilog' if log_scale else 'Linear')
    os.makedirs(output_dir2, exist_ok=True)

    stages = df_stim_nostim[['b', 't']].drop_duplicates().values
    best_protocols = []

    for b, t in stages:
        subset = df_stim_nostim[
            (df_stim_nostim['b'] == b) & (df_stim_nostim['t'] == t)
        ].sort_values('distance', ascending=True).reset_index(drop=True)

        nostim_row  = df_nostim_nostim[(df_nostim_nostim['b'] == b) & (df_nostim_nostim['t'] == t)]
        nostim_dist = nostim_row['distance'].values[0] if len(nostim_row) > 0 else np.nan

        best_row = subset.iloc[0]
        best_protocols.append({
            'b': b, 't': t,
            'sp': best_row['sp'], 'f': best_row['f'], 'inte': best_row['inte'],
            'distance': best_row['distance']
        })

        # prendi una barra ogni 10, più la prima (migliore)
        indices = sorted(set([0] + list(range(0, len(subset), 5))))
        subset_plot = subset.iloc[indices].reset_index(drop=True)

        n          = len(subset_plot)
        distances  = subset_plot['distance'].values
        bar_colors = ['#378ADD' if d < nostim_dist else '#E24B4A' for d in distances]

        # etichetta solo per la prima barra
        #labels = [''] * n
        #labels[0] = f"sp{int(best_row['sp'])} {best_row['f']} {best_row['inte']}"
        labels = [f"sp{int(r.sp)} {r.f} {r.inte}" for _, r in subset_plot.iterrows()]

        fig, ax = plt.subplots(figsize=(max(10, n * 0.5), 10))

        ax.bar(range(n), distances, color=bar_colors, width=0.8, edgecolor='none')
        ax.bar(0, distances[0], color=bar_colors[0], width=0.8, edgecolor='black', linewidth=1.2)

        if not np.isnan(nostim_dist):
            ax.axhline(nostim_dist, color='black', linestyle='--', linewidth=1.2)

        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=22)
        ax.set_xlim(-0.5, n - 0.5)

        if log_scale:
            ax.set_yscale('log')

        ax.set_ylabel('Distance', fontsize=22)
        #ax.set_title(f'Stimulation stage (abeta, tau) = ({b}, {t})', fontsize=25)
        # Modifica i TICK SECONDARI (es. 1.2, 1.4, 1.6...)
        from matplotlib.ticker import NullLocator, MaxNLocator
        # 1. Togli i minor ticks dall'asse X
        ax.yaxis.set_minor_locator(NullLocator())

        # 2. Specifica quanti major ticks al massimo vuoi vedere (es. 5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=2))
        ax.tick_params(axis='y', which='minor', labelsize=22, length=12, width=3)

        # Per l'asse X (che non è logaritmico, ha solo i major) basta questo:
        ax.tick_params(axis='y', which='major', labelsize=22, length=12, width=3)
        ax.tick_params(axis='x', labelsize=22, length=12, width=3)

        legend_handles = [
            mpatches.Patch(color='#378ADD', label='Improvement'),
            mpatches.Patch(color='#E24B4A', label='Worsening'),
        ]
        if not np.isnan(nostim_dist):
            legend_handles.append(
                plt.Line2D([0], [0], color='black', linestyle='--', label=f'No stim: {nostim_dist:.4f}')
            )
        ax.legend(handles=legend_handles, fontsize=22, loc='upper left')

        plt.tight_layout()
        fname = os.path.join(output_dir2, f"dist_b{b}_t{t}.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Salvato: {fname}")
    return pd.DataFrame(best_protocols)
    #df_best_protocols = pd.DataFrame(best_protocols)
    #df_best_protocols.to_pickle(r"..\Tms_data_analysis\Plots\Plot_new_var5\df_best_protocols")

def plot_pre_post_best(path_best_protocols = r"..\Tms_data_analysis\Plots\Plot_new_var5\df_best_protocols"):
    df_best_protocols = pd.read_pickle(path_best_protocols)
    def fmt_inte(x):
        return str(int(x)) if x == int(x) else str(x)
    for _, row in df_best_protocols.iterrows():
        b, t = row['b'], row['t']
        sp   = df_best_protocols.query("b==@b and t==@t")['sp'].values[0]
        f    = df_best_protocols.query("b==@b and t==@t")['f'].values[0]
        inte = df_best_protocols.query("b==@b and t==@t")['inte'].values[0]
        pre_stim = np.load(rf"..\Results_data\Stages_simnibs_2\Stage_{t}_{b}\tau{t}_abeta{b}_a0.0712_b0.0547_g5.2173_v64.22_n0.0407_simtime20000_rs0\PSD_TAVG_tau_{t}_abeta_{b}.npz")
        pre_stim_psd = pre_stim['psd_mean']
        pre_stim_freq = pre_stim['freqs']
        plt.plot(pre_stim_freq, pre_stim_psd)
        plt.xlim(0,45)
        plt.title(f'Stage ({b},{t}) pre stimulation')
        plt.grid()
        output_dir = r"..\Tms_data_analysis\Plots\Plot_new_var5"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(rf"{output_dir}\tavg_psd_pre_b{b}_t{t}.png")
        plt.close()
        post_stim = np.load(rf"..\Results_data\TMS_n02_th18_new_var5\Ab{b}t{t}s{sp}f--{f}_fi{fmt_inte(inte)}\tau{t}_abeta{b}_rs0\PSD_TAVG_tau_{t}_abeta_{b}.npz")
        post_stim_psd = post_stim['psd_mean']
        post_stim_freq = post_stim['freqs']
        plt.plot(post_stim_freq, post_stim_psd)
        plt.title(f'Stage ({b},{t}) post stimulation')
        plt.xlim(0,45)
        plt.grid()
        plt.savefig(rf"{output_dir}\tavg_psd_post_b{b}_t{t}.png")
        plt.close()

# carica i csv
#df_stim   = pd.read_pickle(r"..\Tms_data_analysis\df_distance_stim_new_var5_nostim")
df_stim   = pd.read_pickle(r"..\Tms_data_analysis\Fold4\df_distance_stim_n02_th10_4_nostim_no_exp")
df_nostim = pd.read_pickle(r"..\Tms_data_analysis\df_distance_nostim_nostim")

plot_distance_histograms(df_stim, df_nostim, output_dir=r"..\Tms_data_analysis\Plots\Plot_n02_th10_4_no_exp", log_scale=True)
#plot_pre_post_best()
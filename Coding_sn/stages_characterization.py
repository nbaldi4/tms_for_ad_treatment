import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import colors

from pathlib import Path

from specparam import SpectralModel

def find_2_peaks2(psd_mean, freqs, freq_range = [1, 15]):

    """
    input: psd, freqs, freq range in which peaks are searched
    output: best two peaks in freq range chosen
    """
    fm = SpectralModel(
        peak_width_limits=[0.25, 4.0],  # width max 4 Hz
        max_n_peaks=5,
        min_peak_height=-np.inf,
        verbose = False
    )

    fm.fit(freqs, psd_mean, freq_range)
    #fm.plot(save_fig=True, file_name='fooof_fit.png', file_path='.')

    peaks = fm.results.get_results().peak_fit  # array [CF, PW, BW]

    band1 = peaks[(peaks[:, 0] >= 0) & (peaks[:, 0] < 6)]
    band2 = peaks[(peaks[:, 0] >= 6) & (peaks[:, 0] <= 13)]

    def get_best_peak(band):
        if len(band) == 0:
            return np.array([0.1, 0.1, 0.1])
        return band[np.argmax(band[:, 1])]  # massimo PW

    peak1 = get_best_peak(band1)
    peak2 = get_best_peak(band2)

    return np.vstack([peak1, peak2])

def find_2_peaks(psd_mean, freqs, freq_range=[1, 15], mid_freq = 6):
    fm = SpectralModel(
        peak_width_limits=[1.0, 4.0],
        max_n_peaks=5,
        min_peak_height=0,
        verbose=False
    )
    fm.fit(freqs, psd_mean, freq_range)
    peaks = fm.results.get_results().peak_fit

    band1 = peaks[(peaks[:, 0] >= 0.5) & (peaks[:, 0] < mid_freq)]
    band2 = peaks[(peaks[:, 0] >= mid_freq) & (peaks[:, 0] <= 13)]

    def get_best_peak(band, band_range):
        if len(band) > 0:
            return band[np.argmax(band[:, 1])]  # PW in log10, nativo FOOOF

        # fallback: converti la PSD grezza in log10 per coerenza
        mask     = (freqs >= band_range[0]) & (freqs < band_range[1])
        f_band   = freqs[mask]
        psd_band = np.log10(psd_mean[mask])   # stessa scala di FOOOF
        idx_max  = np.argmax(psd_band)
        return np.array([f_band[idx_max], psd_band[idx_max], np.nan])

    peak1 = get_best_peak(band1, [0.5, mid_freq])
    peak2 = get_best_peak(band2, [mid_freq, 13])

    return np.vstack([peak1, peak2])

def find_2_peaks3(psd_mean, freqs, freq_range=[1, 15], scale='linear'):
    """
    scale = 'log'    → tutto in dB      (PW FOOOF nativo, fallback convertito)
    scale = 'linear' → tutto in lineare (fallback nativo, FOOOF convertito)
    """
    fm = SpectralModel(
        peak_width_limits=[1.0, 4.0],
        max_n_peaks=5,
        min_peak_height=0,
        verbose=False
    )
    fm.fit(freqs, psd_mean, freq_range)
    peaks = fm.results.get_results().peak_fit

    band1 = peaks[(peaks[:, 0] >= 0) & (peaks[:, 0] < 8)]
    band2 = peaks[(peaks[:, 0] >= 8) & (peaks[:, 0] <= 13)]

    def convert_pw(pw_log):
        """FOOOF PW (log10) → lineare"""
        return 10 ** pw_log  # ✅

    def get_best_peak(band, band_range):
        if len(band) > 0:
            best = band[np.argmax(band[:, 1])]
            pw   = best[1] if scale == 'log' else convert_pw(best[1])
            return np.array([best[0], pw, best[2]])

        # fallback
        mask     = (freqs >= band_range[0]) & (freqs < band_range[1])
        f_band   = freqs[mask]
        psd_band = psd_mean[mask]
        idx_max  = np.argmax(psd_band)
        pw_lin   = psd_band[idx_max]
        pw       = np.log10(pw_lin) if scale == 'log' else pw_lin  # ✅
        return np.array([f_band[idx_max], pw, np.nan])

    peak1 = get_best_peak(band1, [0, 8])
    peak2 = get_best_peak(band2, [8, 13])

    return np.vstack([peak1, peak2])

"""
    padded = np.full((2, 3), np.nan)  # nan = picco mancante
    padded[:len(peaks), :] = peaks

    return padded
"""
    

def peak_df_stim(
    root,       # base folder
    dir_pattern # pattern to be redirected to PSD_TAVG file location
):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)

    psd_pattern = re.compile(r"PSD_TAVG_tau_(\d+)_abeta_(\d+)\.npz")

    records = []

    for dirpath, dirnames, filenames in os.walk(root):
        match = re.search(dir_pattern, dirpath)
        if not match:
            continue

        record = {
            'b':        int(match.group(1)),
            't':        int(match.group(2)),
            'sp':       int(match.group(3)),
            'f':        match.group(4),
            'inte':     float(match.group(5)),
            'tau':      int(match.group(6)),
            'abeta':    int(match.group(7)),
            'rs':       int(match.group(8)),
            'freqs':    None,
            'psd_mean': None,
        }

        for fname in filenames:
            if psd_pattern.match(fname):
                fpath = os.path.join(dirpath, fname)
                with np.load(fpath) as f:
                    record['freqs']    = f['freqs']
                    record['psd_mean'] = f['psd_mean']
                break

        if record['freqs'] is None:
            continue

        records.append(record)

    df = pd.DataFrame(records).drop_duplicates(
        subset=['b', 't', 'sp', 'f', 'inte', 'tau', 'abeta', 'rs']
    )

    records_hl = []
    for _, row in df.iterrows():
        peaks = find_2_peaks(freqs=row['freqs'], psd_mean=row['psd_mean'])

        records_hl.append({
            'b':        row['b'],
            't':        row['t'],
            'sp':       row['sp'],
            'f':        row['f'],
            'inte':     row['inte'],
            'rs':       row['rs'],
            'PSD_peak_1': peaks[0],   # array [CF, PW, BW] o [nan, nan, nan]
            'PSD_peak_2': peaks[1],
        })

    return pd.DataFrame(records_hl)

def peak_df_nostim(
    root,       # base folder
    dir_pattern # pattern to be redirected to PSD_TAVG file location
):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)

    psd_pattern = re.compile(r"PSD_TAVG_tau_(\d+)_abeta_(\d+)\.npz")

    records = []

    for dirpath, dirnames, filenames in os.walk(root):
        match = re.search(dir_pattern, dirpath)
        if not match:
            continue

        record = {
            't':        int(match.group(1)),
            'b':        int(match.group(2)),
            'rs':       int(match.group(5)),
            'freqs':    None,
            'psd_mean': None,
        }

        for fname in filenames:
            if psd_pattern.match(fname):
                fpath = os.path.join(dirpath, fname)
                with np.load(fpath) as f:
                    record['freqs']    = f['freqs']
                    record['psd_mean'] = f['psd_mean']
                break

        if record['freqs'] is None:
            continue

        records.append(record)

    df = pd.DataFrame(records).drop_duplicates(
        subset=['b', 't', 'rs']
    )

    records_hl = []
    for _, row in df.iterrows():
        peaks = find_2_peaks(freqs=row['freqs'], psd_mean=row['psd_mean'])

        records_hl.append({
            'b':        row['b'],
            't':        row['t'],
            'rs':       row['rs'],
            'PSD_peak_1': peaks[0],   # array [CF, PW, BW] o [nan, nan, nan]
            'PSD_peak_2': peaks[1],
        })

    return pd.DataFrame(records_hl)

def mean_peaks(df):
    for i, col in enumerate(['CF', 'PW', 'BW']):
        df[f'peak1_{col}'] = df['PSD_peak_1'].apply(lambda x: x[i])
        df[f'peak2_{col}'] = df['PSD_peak_2'].apply(lambda x: x[i])

    df_mean_peak = (
        df
        .groupby(['b', 't'])[['peak1_CF','peak1_PW','peak1_BW',
                               'peak2_CF','peak2_PW','peak2_BW']]
        .mean()
        .reset_index()
    )

    df_mean_peak['PSD_peak_1'] = df_mean_peak[['peak1_CF','peak1_PW','peak1_BW']].values.tolist()
    df_mean_peak['PSD_peak_2'] = df_mean_peak[['peak2_CF','peak2_PW','peak2_BW']].values.tolist()

    return df_mean_peak[['b', 't', 'PSD_peak_1', 'PSD_peak_2']]

df_stim = peak_df_stim(
    root=r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\TMS_n02_th18_new_var5",
    dir_pattern=r"Ab(\d+)t(\d+)s(\d+)f--(\w+)_fi(\d*\.?\d+)[/\\]tau(\d+)_abeta(\d+).*_rs(\d+)"
)
df_stim.to_pickle(r"C:\Users\User\OneDrive - University of Pisa\Desktop\df_stim_prova")


df_nostim = peak_df_nostim(
    root=r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Stages_simnibs_2",
    dir_pattern=r"Stage_(\d+)_(\d+)[/\\]tau(\d+)_abeta(\d+)_a0.0712_b0.0547_g5.2173_v64.22_n0.0407_simtime20000_rs(\d+)"
)
df_nostim.to_pickle(r"C:\Users\User\OneDrive - University of Pisa\Desktop\df_nostim_prova")

df_nostim_mean = mean_peaks(df_nostim)
df_nostim_mean.to_pickle(r"C:\Users\User\OneDrive - University of Pisa\Desktop\df_nostim_mean_prova")

def compute_ratio(peak1, peak2):
    """Rapporto PW_peak2 / PW_peak1 (alpha / theta-delta)"""
    pw1 = peak1[1] if not np.isnan(peak1[1]) else np.nan
    pw2 = peak2[1] if not np.isnan(peak2[1]) else np.nan
    return pw2 / pw1 if pw1 != 0 else np.nan


def add_ratio(df):
    """Aggiunge colonna 'ratio' a qualsiasi df con PSD_peak_1 e PSD_peak_2"""
    df = df.copy()
    df['ratio'] = df.apply(
        lambda row: compute_ratio(row['PSD_peak_1'], row['PSD_peak_2']), axis=1
    )
    return df


def compute_distance2(df_stim, df_nostim_mean):
    """
    Per ogni riga dello stimolato, calcola:
        distance = ratio_stim - ratio_nostim
    dove ratio_nostim è quello della coppia (b, t) corrispondente.
    """
    df_stim   = add_ratio(df_stim)
    df_nostim_mean = add_ratio(df_nostim_mean)

    # rinomina per il merge
    df_ref = df_nostim_mean[['b', 't', 'ratio']].rename(columns={'ratio': 'ratio_nostim'})

    df_out = df_stim.merge(df_ref, on=['b', 't'], how='left')
    df_out['distance'] = df_out['ratio'] - df_out['ratio_nostim']

    return df_out


def compute_distance(df_stim, df_nostim_mean):
    df_stim        = add_ratio(df_stim)
    df_nostim_mean = add_ratio(df_nostim_mean)

    # prendi solo lo stadio sano (0, 0)
    ratio_healthy = df_nostim_mean.loc[
        (df_nostim_mean['b'] == 0) & (df_nostim_mean['t'] == 0), 'ratio'
    ].values[0]

    df_out = df_stim.copy()
    df_out['ratio_nostim'] = ratio_healthy
    df_out['distance']     = df_out['ratio'] - ratio_healthy

    return df_out

df_result_stim = compute_distance(df_stim, df_nostim_mean)
df_result_stim.to_pickle(r"C:\Users\User\OneDrive - University of Pisa\Desktop\df_distance_stim_nostim")
df_result_nostim = compute_distance(df_nostim_mean, df_nostim_mean)
df_result_nostim.to_pickle(r"C:\Users\User\OneDrive - University of Pisa\Desktop\df_distance_nostim_nostim")
"""
stages_peaks = {}  # dizionario per accedere i file per (x, y, z)
mean_stages_peaks = {}  # dizionario per accedere i file per (x, y, z)

tau_max = 6      # 0-5
abeta_max = 6    # 0-5
seed_max = 20    # 0-19

for x in range(tau_max):
    for y in range(abeta_max):
        for z in range(seed_max):"""
#            path = base_dir / \
#                   f"Stage_{x}_{y}" / \
#                   f"tau{x}_abeta{y}_a0.0712_b0.0547_g5.2173_v64.22_n0.0407_simtime20000_rs{z}" / \
#                   f"PSD_TAVG_tau_{x}_abeta_{y}.npz"
"""
            if path.exists():
                file = np.load(path)
                freqs    = file['freqs']
                psd_mean = file['psd_mean']
                peaks = find_2_peaks(psd_mean=psd_mean, freqs=freqs)
                stages_peaks[(x, y, z)] = peaks
                #print(f"Stage abeta {y} tau {x} random seed {z}")            

for x in range(tau_max):
    for y in range(abeta_max):
        # Raccogli i peaks per tutti i seeds disponibili
        peaks_per_seed = []
        for z in range(seed_max):
            if (x, y, z) in stages_peaks:
                peaks_per_seed.append(stages_peaks[(x, y, z)])
        
        # peaks_per_seed è una lista di array, ognuno con i parametri dei picchi
        # Converti in array e fai la media su z (axis=0)
        if len(peaks_per_seed) > 0:
            peaks_array = np.array(peaks_per_seed)  # shape: (n_seeds, n_peaks, 3)
            mean_stages_peaks[(x, y)] = np.nanmean(peaks_array, axis=0)  # media sui seeds
        
        if x == 0 and y == 0:
            print(f"Stage abeta {y} tau {x} mean above seeds")
            print(mean_stages_peaks[(x, y)])
            """
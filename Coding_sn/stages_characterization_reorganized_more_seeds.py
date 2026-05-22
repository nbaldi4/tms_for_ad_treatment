import os
import re

import pandas as pd
import numpy as np


def find_2_peaks(psd_mean, freqs, freq_range=[1, 15], mid_freq=6):

    def get_best_peak(band_range):
        mask     = (freqs >= band_range[0]) & (freqs < band_range[1])
        f_band   = freqs[mask]
        psd_band = psd_mean[mask]
        idx_max  = np.argmax(psd_band)
        return np.array([f_band[idx_max], psd_band[idx_max], np.nan])

    peak1 = get_best_peak([freq_range[0], mid_freq])
    peak2 = get_best_peak([mid_freq, freq_range[1]])

    return np.vstack([peak1, peak2])


def peak_df_stim(root, dir_pattern):
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
            'b':          row['b'],
            't':          row['t'],
            'sp':         row['sp'],
            'f':          row['f'],
            'inte':       row['inte'],
            'rs':         row['rs'],
            'PSD_peak_1': peaks[0],
            'PSD_peak_2': peaks[1],
        })

    return pd.DataFrame(records_hl)


def peak_df_nostim(root, dir_pattern):
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

    df = pd.DataFrame(records).drop_duplicates(subset=['b', 't', 'rs'])

    records_hl = []
    for _, row in df.iterrows():
        peaks = find_2_peaks(freqs=row['freqs'], psd_mean=row['psd_mean'])
        records_hl.append({
            'b':          row['b'],
            't':          row['t'],
            'rs':         row['rs'],
            'PSD_peak_1': peaks[0],
            'PSD_peak_2': peaks[1],
        })

    return pd.DataFrame(records_hl)


def mean_peaks(df):
    for i, col in enumerate(['CF', 'PW', 'BW']):
        df[f'peak1_{col}'] = df['PSD_peak_1'].apply(lambda x: x[i])
        df[f'peak2_{col}'] = df['PSD_peak_2'].apply(lambda x: x[i])

    df_mean_peak = (
        df
        .groupby(['b', 't'])[['peak1_CF', 'peak1_PW', 'peak1_BW',
                               'peak2_CF', 'peak2_PW', 'peak2_BW']]
        .mean()
        .reset_index()
    )

    df_mean_peak['PSD_peak_1'] = df_mean_peak[['peak1_CF', 'peak1_PW', 'peak1_BW']].values.tolist()
    df_mean_peak['PSD_peak_2'] = df_mean_peak[['peak2_CF', 'peak2_PW', 'peak2_BW']].values.tolist()

    return df_mean_peak[['b', 't', 'PSD_peak_1', 'PSD_peak_2']]

def mean_peaks_stim(df):
    """Media dei picchi sui seed (rs) a parità di b, t, sp, f, inte."""
    for i, col in enumerate(['CF', 'PW', 'BW']):
        df[f'peak1_{col}'] = df['PSD_peak_1'].apply(lambda x: x[i])
        df[f'peak2_{col}'] = df['PSD_peak_2'].apply(lambda x: x[i])

    group_keys = ['b', 't', 'sp', 'f', 'inte']
    value_cols = ['peak1_CF', 'peak1_PW', 'peak1_BW',
                  'peak2_CF', 'peak2_PW', 'peak2_BW']

    df_mean = (
        df
        .groupby(group_keys)[value_cols]
        .mean()
        .reset_index()
    )

    df_mean['PSD_peak_1'] = df_mean[['peak1_CF', 'peak1_PW', 'peak1_BW']].values.tolist()
    df_mean['PSD_peak_2'] = df_mean[['peak2_CF', 'peak2_PW', 'peak2_BW']].values.tolist()

    return df_mean[group_keys + ['PSD_peak_1', 'PSD_peak_2']]


def compute_ratio(peak1, peak2):
    pw1 = peak1[1] if not np.isnan(peak1[1]) else np.nan
    pw2 = peak2[1] if not np.isnan(peak2[1]) else np.nan
    #return (10**pw2) / (10**pw1) if pw1 != 0 else np.nan
    return (pw2) / (pw1) if pw1 != 0 else np.nan

def add_ratio(df):
    df = df.copy()
    df['ratio'] = df.apply(
        lambda row: compute_ratio(row['PSD_peak_1'], row['PSD_peak_2']), axis=1
    )
    return df


def compute_distance(df_stim, df_nostim_mean):
    df_stim        = add_ratio(df_stim)
    df_nostim_mean = add_ratio(df_nostim_mean)

    ratio_healthy = df_nostim_mean.loc[
        (df_nostim_mean['b'] == 0) & (df_nostim_mean['t'] == 0), 'ratio'
    ].values[0]

    df_out = df_stim.copy()
    df_out['ratio_nostim'] = ratio_healthy
    df_out['distance']     = np.abs(df_out['ratio'] - ratio_healthy)

    return df_out


# --- Esecuzione ---
os.makedirs(r"..\Tms_data_analysis\Fold4", exist_ok=True)
df_stim = peak_df_stim(
    root=r"..\Results_data\n02_th10_4",
    dir_pattern=r"Ab(\d+)t(\d+)s(\d+)f--(\w+)_fi(\d*\.?\d+)[/\\]tau(\d+)_abeta(\d+).*_rs(\d+)"
)
df_stim.to_pickle(r"..\Tms_data_analysis\Fold4\df_stim_n02_th10_4_no_exp")
df_stim.to_csv(r"..\Tms_data_analysis\Fold4\df_stim_n02_th10_4_no_exp_cvs")

# media sui seed prima di calcolare il rapporto
df_stim_mean = mean_peaks_stim(df_stim)
df_stim_mean.to_pickle(r"..\Tms_data_analysis\Fold4\df_stim_mean_n02_th10_4_no_exp")
df_stim_mean.to_csv(r"..\Tms_data_analysis\Fold4\df_stim_mean_n02_th10_4_no_exp_cvs")

df_nostim = peak_df_nostim(
    root=r"..\Results_data\Stages_simnibs_2",
    dir_pattern=r"Stage_(\d+)_(\d+)[/\\]tau(\d+)_abeta(\d+)_a0.0712_b0.0547_g5.2173_v64.22_n0.0407_simtime20000_rs(\d+)"
)
df_nostim.to_pickle(r"..\Tms_data_analysis\df_nostim")
df_nostim.to_csv(r"..\Tms_data_analysis\df_nostim_cvs")

df_nostim_mean = mean_peaks(df_nostim)
df_nostim_mean.to_pickle(r"..\Tms_data_analysis\df_nostim_mean")
df_nostim_mean.to_csv(r"..\Tms_data_analysis\df_nostim_mean_cvs")

df_result_stim = compute_distance(df_stim_mean, df_nostim_mean)
#df_result_stim = compute_distance(df_stim, df_nostim_mean)
df_result_stim.to_pickle(r"..\Tms_data_analysis\Fold4\df_distance_stim_n02_th10_4_nostim_no_exp")
df_result_stim.to_csv(r"..\Tms_data_analysis\Fold4\df_distance_stim_n02_th10_4_nostim_no_exp_cvs")

df_result_nostim = compute_distance(df_nostim_mean, df_nostim_mean)
df_result_nostim.to_pickle(r"..\Tms_data_analysis\df_distance_nostim_nostim")
df_result_nostim.to_csv(r"..\Tms_data_analysis\df_distance_nostim_nostim_cvs")
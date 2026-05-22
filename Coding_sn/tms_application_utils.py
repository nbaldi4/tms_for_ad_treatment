"""
Visualization, simulation, and signal processing utilities for brain activity modeling
--------------------------------------------------------------------------------------

This module provides a set of helper functions for:
- Visualizing brain surfaces, sensors, and connectivity matrices
- Simulating brain activity using the Jansen–Rit model within The Virtual Brain (TVB)
- Preprocessing EEG and region-based signals
- Computing and plotting functional connectivity and power spectral density (PSD)
"""
import os
import re

from tvb.simulator.lab import *
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import global_variable_creation as gv
from utils import make_train, new_parcellation_simulate
from classes import MultiStimuliRegion
import logging

def stimulation_analysis(times, timeseries, stimulus_onset, preprocess_window=1000):
    mask_pre_stimulus = (times < (stimulus_onset - 100))
    mask_post_stimulus = (times >= stimulus_onset)
    assert(stimulus_onset >= preprocess_window + 1000)
    timeseries_mask_pre_stimulus = timeseries[mask_pre_stimulus]
    timeseries_mask_post_stimulus = timeseries[mask_post_stimulus]
    timeseries_baseline = np.mean(timeseries_mask_pre_stimulus)
    pre_stimulus_amplitude = np.max(np.abs(timeseries_mask_pre_stimulus - timeseries_baseline))
    post_stimulus_amplitude = np.max(np.abs(timeseries_mask_post_stimulus - timeseries_baseline))
    return timeseries_baseline, pre_stimulus_amplitude, post_stimulus_amplitude

def tms_pulse_kick_all_channels(scale_fact, tau, abeta, seed, period, weights_vector, stimulus_onset, freq):
    """
    Simulation for given scale_factor and stimulus
    Return: (baseline, pre_amp, post_amp, sf, seed, channel)
    """
    print('abeta', abeta)
    print('tau',tau)
    for name in logging.root.manager.loggerDict:
        if name.startswith('tvb'):
            logging.getLogger(name).setLevel(logging.ERROR)
            logging.getLogger(name).handlers = []

    gv.random_seed = seed
    sim_time = stimulus_onset + 2000 + 500

    train = make_train(
        np.arange(0, 86),
        scale_fact * weights_vector,
        onset=stimulus_onset,
        T=period,
        tau=1.0
    )

    stimulus = MultiStimuliRegion(train)
    stimulus.configure_space()
    time = np.r_[stimulus_onset - 500:stimulus_onset + 500 + 2000:0.5]
    stimulus.configure_time(time)
    pattern = stimulus()


    ttavg_stim, tavg_healthy_stim, teeg_stim, eeg_healthy_stim = new_parcellation_simulate(
        g=gv.coup,
        velocity=gv.vel,
        noise=gv.noi,
        a=gv.aa,
        b=gv.bb,
        sim_time=sim_time,
        stimulus=stimulus,
        tau_stage=tau,
        abeta_stage=abeta,
        freq = freq
    )

    ttavg_stim = ttavg_stim[1000:]
    tavg_healthy_stim = tavg_healthy_stim[1000:, :, :, :]
    teeg_stim = teeg_stim[1000:]
    eeg_healthy_stim = eeg_healthy_stim[1000:, :, :, :]

    signal = tavg_healthy_stim
    time_array = ttavg_stim

    # array shape: (time, 1, channels, 1)
    n_channels = signal.shape[2]

    rows = []
    for channel in range(n_channels):
        baseline, pre_amp, post_amp = stimulation_analysis(
            times=time_array,
            timeseries=signal[:, 0, channel, 0],
            stimulus_onset=stimulus_onset
        )
        rows.append((baseline, pre_amp, post_amp, scale_fact, gv.random_seed, channel))

    return rows

def tms_seeds_analysis(file_name, tau, abeta, period, weights_vector, stimulus_onset, freq, tested_values=100, n_seeds=20):

    #scale_factor_span = np.logspace(0, 3, tested_values)#np.logspace(-1, 2.5, tested_values)
    scale_factor_span = np.array([0.5,1,5,10,15,20]) #np.array([0.5,2,4,6,8])
    seed_span = np.arange(n_seeds)  # 0 ... 19

    # Genera tutte le coppie (sf, seed)
    pairs = [(sf, seed) for sf in scale_factor_span for seed in seed_span]

    # Una simulazione per coppia (sf, seed), parallelizzata
    all_results = Parallel(n_jobs=10)(
        delayed(tms_pulse_kick_all_channels)(
            stimulus_onset=stimulus_onset,
            period=period,
            tau=tau,
            abeta=abeta,
            weights_vector=weights_vector,
            scale_fact=sf,
            seed=seed,
            freq=freq
        )
        for sf, seed in pairs
    )

    # Appiattisci: lista di liste → lista di righe
    rows = [row for sim_rows in all_results for row in sim_rows]

    # Costruisci DataFrame e salva
    df = pd.DataFrame(rows, columns=["baseline", "amplitude_pre_stimulus", "amplitude_post_stimulus", "sf", "seed", "channel"])
    df["channel"] = df["channel"].astype(int)
    df["seed"] = df["seed"].astype(int)

    #df.to_csv("tavg_protocollo_1.csv", index=False)
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    df.to_pickle(file_name)

    #print(f"Completed: {tested_values} sf × {n_seeds} seeds × {df['channel'].nunique()} channels = {len(df)} total rows.")
    return df.to_numpy()

def check_majority_seeds(df, ratio_threshold):
    """
    For every couple (sf, channel), we check if at least threshold of
    seeds satisfies is_greater condition 
    
    Returns:
        Pandas DataFrame with columns: sf, channel, fraction_greater, majority_satisfied
    """
    result = (
        df.groupby(["sf", "channel"])["is_greater"]
        .mean()
        .reset_index()
        .rename(columns={"is_greater": "fraction_greater"})
    )
    result["majority_satisfied"] = result["fraction_greater"] >= ratio_threshold
    return result

def best_sf(df, amplitude_threshold_up=1.5, ampltitude_threshold_down=0.5, ratio_threshold=0.5):
    df["is_greater"] = (df["amplitude_post_stimulus"] > (amplitude_threshold_up * df["amplitude_pre_stimulus"]))#|(df["amplitude_post_stimulus"] < (ampltitude_threshold_down * df["amplitude_pre_stimulus"]))
    majority_result = check_majority_seeds(df, ratio_threshold)
    
    channels_per_sf = (
        majority_result
        .groupby("sf")["majority_satisfied"]
        .sum()
        .reset_index()
        .rename(columns={"majority_satisfied": "n_channels_satisfied"})
    )
    
    best = channels_per_sf.loc[channels_per_sf["n_channels_satisfied"].idxmax()]
    best_sf_value = best["sf"]
    best_n_channels = best["n_channels_satisfied"]

    # Raggruppa per (sf, channel) invece di filtrare su best_sf_value
    grouped = df.groupby(["sf", "channel"])

    bool_per_channel = (
        grouped["is_greater"]
        .mean()
        .ge(ratio_threshold)
    )

    mean_post_amplitude = grouped["amplitude_post_stimulus"].mean()
    mean_pre_amplitude  = grouped["amplitude_pre_stimulus"].mean()

    result = pd.DataFrame({
        "is_greater"   : bool_per_channel,
        "mean_amp_post": mean_post_amplitude,
        "mean_amp_pre" : mean_pre_amplitude,
        "mean_baseline": grouped["baseline"].mean(),
    }).reset_index()  # porta sf e channel come colonne esplicite

    return result, best_sf_value, best_n_channels

def infer_stage2(root = r"..\tms_for_ad_treatment\Results_data\Stat_an", dir_pattern = r"St_b(\d+)t(\d+)_sp(\d+)_f--(\w+)_f[/\\]tau(\d+)_abeta(\d+).*_rs(\d+)"):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)

    broad_pattern = re.compile(r"FC_TAVG_tau_(\d+)_abeta_(\d+)_band_([\d\.\-]+)_Hz\.npz")
    fc_pattern  = re.compile(r"FC_TAVG (\w+)_tau_(\d+)_abeta_(\d+)_band_([\d\.\-]+)_Hz\.npz")
    psd_pattern = re.compile(r"PSD_TAVG_tau_(\d+)_abeta_(\d+)\.npz")

    bands = ['alpha', 'delta', 'theta', 'beta', 'broad']

    records = []

    for dirpath, dirnames, filenames in os.walk(root):
        match = re.search(dir_pattern, dirpath)
        if match:
            record = {
                'b':        int(match.group(1)),
                't':        int(match.group(2)),
                'sp':       int(match.group(3)),
                'f':        match.group(4),
                'tau':      int(match.group(5)),
                'abeta':    int(match.group(6)),
                'rs':       int(match.group(7)),
                'path':     dirpath,
                'PSD_TAVG': None,
            }
            for band in bands:
                record[f'FC_{band}'] = None

            for fname in filenames:
                fpath = os.path.join(dirpath, fname)

                if psd_pattern.match(fname):
                    record['PSD_TAVG'] = np.load(fpath)

                if broad_pattern.match(fname):
                    record['FC_broad'] = np.load(fpath)

                fc_match = fc_pattern.match(fname)
                if fc_match:
                    band = fc_match.group(1)
                    if band in bands and band != 'broad':
                        record[f'FC_{band}'] = np.load(fpath)

            records.append(record)

    df = pd.DataFrame(records).drop_duplicates(subset=['b', 't', 'sp', 'f', 'tau', 'abeta', 'rs'])

    #print(df)
    #print(f"\nTotale combinazioni trovate: {len(df)}")

    # Estrai array dai file npz
    df['freqs']    = df['PSD_TAVG'].apply(lambda x: x['freqs']        if x is not None else None)
    df['psd_mean'] = df['PSD_TAVG'].apply(lambda x: x['psd_mean']     if x is not None else None)
    for band in bands:
        df[f'FC_{band}_normal'] = df[f'FC_{band}'].apply(lambda x: x['normal_size'] if x is not None else None)

    # Calcola highlights
    records_hl = []
    for _, row in df.iterrows():
        freqs    = row['freqs']
        psd_mean = row['psd_mean']

        psd_broad = np.sum(psd_mean[(freqs >= 0.5) & (freqs < 45)])
        psd_delta = np.sum(psd_mean[(freqs >= 0.5) & (freqs < 4)])
        psd_theta = np.sum(psd_mean[(freqs >= 4)   & (freqs < 8)])
        psd_alpha = np.sum(psd_mean[(freqs >= 8)   & (freqs < 13)])
        psd_beta  = np.sum(psd_mean[(freqs >= 13)  & (freqs < 30)])

        records_hl.append({
            'b'                : row['b'],
            't'                : row['t'],
            'sp'               : row['sp'],
            'f'                : row['f'],
            'rs'               : row['rs'],
            'FC_abs_mean_broad': np.mean(np.abs(row['FC_broad_normal'])) if row['FC_broad_normal'] is not None else None,
            'FC_abs_mean_delta': np.mean(np.abs(row['FC_delta_normal'])) if row['FC_delta_normal'] is not None else None,
            'FC_abs_mean_theta': np.mean(np.abs(row['FC_theta_normal'])) if row['FC_theta_normal'] is not None else None,
            'FC_abs_mean_alpha': np.mean(np.abs(row['FC_alpha_normal'])) if row['FC_alpha_normal'] is not None else None,
            'FC_abs_mean_beta' : np.mean(np.abs(row['FC_beta_normal']))  if row['FC_beta_normal']  is not None else None,
            'PSD_delta_over_br': psd_delta / psd_broad if psd_broad != 0 else None,
            'PSD_theta_over_br': psd_theta / psd_broad if psd_broad != 0 else None,
            'PSD_alpha_over_br': psd_alpha / psd_broad if psd_broad != 0 else None,
            'PSD_beta_over_br' : psd_beta  / psd_broad if psd_broad != 0 else None,
        })

    df_highlights = pd.DataFrame(records_hl)
    return df_highlights

def infer_stage(root = r"..\tms_for_ad_treatment\Results_data\Stat_an", dir_pattern = r"St_b(\d+)t(\d+)_sp(\d+)_f--(\w+)_f[/\\]tau(\d+)_abeta(\d+).*_rs(\d+)"):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)

    broad_pattern = re.compile(r"FC_TAVG_tau_(\d+)_abeta_(\d+)_band_([\d\.\-]+)_Hz\.npz")
    fc_pattern    = re.compile(r"FC_TAVG (\w+)_tau_(\d+)_abeta_(\d+)_band_([\d\.\-]+)_Hz\.npz")
    psd_pattern   = re.compile(r"PSD_TAVG_tau_(\d+)_abeta_(\d+)\.npz")

    bands = ['alpha', 'delta', 'theta', 'beta', 'broad']

    records = []

    for dirpath, dirnames, filenames in os.walk(root):
        match = re.search(dir_pattern, dirpath)
        if match:
            record = {
                'b':     int(match.group(1)),
                't':     int(match.group(2)),
                'sp':    int(match.group(3)),
                'f':     match.group(4),
                'inte':  float(match.group(5)),
                'tau':   int(match.group(6)),
                'abeta': int(match.group(7)),
                'rs':    int(match.group(8)),
                'path':  dirpath,
                'freqs':    None,
                'psd_mean': None,
            }
            for band in bands:
                record[f'FC_{band}_normal'] = None

            for fname in filenames:
                fpath = os.path.join(dirpath, fname)

                # ✅ Usa with: il file viene chiuso subito dopo l'estrazione
                if psd_pattern.match(fname):
                    with np.load(fpath) as f:
                        record['freqs']    = f['freqs']
                        record['psd_mean'] = f['psd_mean']

                if broad_pattern.match(fname):
                    with np.load(fpath) as f:
                        record['FC_broad_normal'] = f['normal_size']

                fc_match = fc_pattern.match(fname)
                if fc_match:
                    band = fc_match.group(1)
                    if band in bands and band != 'broad':
                        with np.load(fpath) as f:
                            record[f'FC_{band}_normal'] = f['normal_size']

            records.append(record)

    df = pd.DataFrame(records).drop_duplicates(subset=['b', 't', 'sp', 'f', 'inte', 'tau', 'abeta', 'rs'])

    # Calcola highlights (ora gli array sono già estratti, non serve più .apply())
    records_hl = []
    for _, row in df.iterrows():
        freqs    = row['freqs']
        psd_mean = row['psd_mean']

        psd_broad = np.sum(psd_mean[(freqs >= 0.5) & (freqs < 45)])
        psd_delta = np.sum(psd_mean[(freqs >= 0.5) & (freqs < 4)])
        psd_theta = np.sum(psd_mean[(freqs >= 4)   & (freqs < 8)])
        psd_alpha = np.sum(psd_mean[(freqs >= 8)   & (freqs < 13)])
        psd_beta  = np.sum(psd_mean[(freqs >= 13)  & (freqs < 30)])

        records_hl.append({
            'b'                : row['b'],
            't'                : row['t'],
            'sp'               : row['sp'],
            'f'                : row['f'],
            'inte'             : row['inte'],
            'rs'               : row['rs'],
            'FC_abs_mean_broad': np.mean(np.abs(row['FC_broad_normal'])) if row['FC_broad_normal'] is not None else None,
            'FC_abs_mean_delta': np.mean(np.abs(row['FC_delta_normal'])) if row['FC_delta_normal'] is not None else None,
            'FC_abs_mean_theta': np.mean(np.abs(row['FC_theta_normal'])) if row['FC_theta_normal'] is not None else None,
            'FC_abs_mean_alpha': np.mean(np.abs(row['FC_alpha_normal'])) if row['FC_alpha_normal'] is not None else None,
            'FC_abs_mean_beta' : np.mean(np.abs(row['FC_beta_normal']))  if row['FC_beta_normal']  is not None else None,
            'PSD_delta_over_br': psd_delta / psd_broad if psd_broad != 0 else None,
            'PSD_theta_over_br': psd_theta / psd_broad if psd_broad != 0 else None,
            'PSD_alpha_over_br': psd_alpha / psd_broad if psd_broad != 0 else None,
            'PSD_beta_over_br' : psd_beta  / psd_broad if psd_broad != 0 else None,
        })

    df_highlights = pd.DataFrame(records_hl)
    return df_highlights

def load_npz(fpath):
    with np.load(fpath) as data:
        return {key: data[key] for key in data.files}

def stages_boxplots(dir_pattern = r"Stage_(\d+)_(\d+)[/\\]tau(\d+)_abeta(\d+).*_rs(\d+)"):
    root = r"..\tms_for_ad_treatment\Results_data\Stages_simnibs_2"
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)

    broad_pattern = re.compile(r"FC_TAVG_tau_(\d+)_abeta_(\d+)_band_([\d\.\-]+)_Hz\.npz")
    fc_pattern  = re.compile(r"FC_TAVG[ _](\w+)_tau_(\d+)_abeta_(\d+)_band_([\d\.\-]+)_Hz\.npz")
    psd_pattern = re.compile(r"PSD_TAVG_tau_(\d+)_abeta_(\d+)\.npz")

    bands = ['alpha', 'delta', 'theta', 'beta', 'broad']

    records = []

    for dirpath, dirnames, filenames in os.walk(root):
        match = re.search(dir_pattern, dirpath)
        if match:
            record = {
                't':        int(match.group(1)),
                'b':        int(match.group(2)),
                'tau':      int(match.group(3)),
                'abeta':    int(match.group(4)),
                'rs':       int(match.group(5)),
                'path':     dirpath,
                'PSD_TAVG': None,
            }
            for band in bands:
                record[f'FC_{band}'] = None

            for fname in filenames:
                fpath = os.path.join(dirpath, fname)

                if psd_pattern.match(fname):
                    record['PSD_TAVG'] = np.load(fpath)

                if broad_pattern.match(fname):
                    record['FC_broad'] = np.load(fpath)

                fc_match = fc_pattern.match(fname)
                if fc_match:
                    band = fc_match.group(1)
                    if band in bands and band != 'broad':
                        record[f'FC_{band}'] = load_npz(fpath)

            records.append(record)

    df = pd.DataFrame(records).drop_duplicates(subset=['b', 't', 'tau', 'abeta', 'rs'])

    #print(df)
    #print(f"\nTotale combinazioni trovate: {len(df)}")
    
    # Estrai array dai file npz
    df['freqs']    = df['PSD_TAVG'].apply(lambda x: x['freqs']        if x is not None else None)
    df['psd_mean'] = df['PSD_TAVG'].apply(lambda x: x['psd_mean']     if x is not None else None)
    for band in bands:
        df[f'FC_{band}_normal'] = df[f'FC_{band}'].apply(lambda x: x['normal_size'] if x is not None else None)

    # Calcola highlights
    records_hl = []
    for _, row in df.iterrows():
        freqs    = row['freqs']
        psd_mean = row['psd_mean']

        psd_broad = np.sum(psd_mean[(freqs >= 0.5) & (freqs < 45)])
        psd_delta = np.sum(psd_mean[(freqs >= 0.5) & (freqs < 4)])
        psd_theta = np.sum(psd_mean[(freqs >= 4)   & (freqs < 8)])
        psd_alpha = np.sum(psd_mean[(freqs >= 8)   & (freqs < 13)])
        psd_beta  = np.sum(psd_mean[(freqs >= 13)  & (freqs < 30)])

        records_hl.append({
            'b'                : row['b'],
            't'                : row['t'],
            'rs'               : row['rs'],
            'FC_abs_mean_broad': np.mean(np.abs(row['FC_broad_normal'])) if row['FC_broad_normal'] is not None else None,
            'FC_abs_mean_delta': np.mean(np.abs(row['FC_delta_normal'])) if row['FC_delta_normal'] is not None else None,
            'FC_abs_mean_theta': np.mean(np.abs(row['FC_theta_normal'])) if row['FC_theta_normal'] is not None else None,
            'FC_abs_mean_alpha': np.mean(np.abs(row['FC_alpha_normal'])) if row['FC_alpha_normal'] is not None else None,
            'FC_abs_mean_beta' : np.mean(np.abs(row['FC_beta_normal']))  if row['FC_beta_normal']  is not None else None,
            'PSD_delta_over_br': psd_delta / psd_broad if psd_broad != 0 else None,
            'PSD_theta_over_br': psd_theta / psd_broad if psd_broad != 0 else None,
            'PSD_alpha_over_br': psd_alpha / psd_broad if psd_broad != 0 else None,
            'PSD_beta_over_br' : psd_beta  / psd_broad if psd_broad != 0 else None,
        })

    df_highlights = pd.DataFrame(records_hl)
    #print(df_highlights.columns)
    #print(df_highlights.query("b==0 and t==1 and sp==0 and rs==0")["FC_abs_mean_alpha"])
    return df_highlights

def no_stim_boxplot_stats(df_highlights):
    metrics = ['Fbr', 'Fd', 'Ft', 'Fa', 'Fbe',
               'Pdobr', 'Ptobr', 'Paobr', 'Pbobr']

    records = []
    for (b, t), group in df_highlights.groupby(['b', 't']):
        record = {'b': b, 't': t}
        for metric in metrics:
            vals = group[metric].dropna()
            record[f'{metric}_min']    = np.percentile(vals, 0)
            record[f'{metric}_q25']    = np.percentile(vals, 25)
            record[f'{metric}_med']    = np.percentile(vals, 50)
            record[f'{metric}_q75']    = np.percentile(vals, 75)
            record[f'{metric}_max']    = np.percentile(vals, 100)
        records.append(record)

    return pd.DataFrame(records)

def stim_boxplot_stats(df_highlights):
    metrics = ['Fbr', 'Fd', 'Ft', 'Fa', 'Fbe',
               'Pdobr', 'Ptobr', 'Paobr', 'Pbobr']

    records = []
    for (b, t, sp, f, intensity), group in df_highlights.groupby(['b', 't', 'sp', 'f', 'inte']):
        record = {'b': b, 't': t, 'sp': sp, 'f': f, 'inte':intensity }
        for metric in metrics:
            vals = group[metric].dropna()
            record[f'{metric}_min']    = np.percentile(vals, 0)
            record[f'{metric}_q25']    = np.percentile(vals, 25)
            record[f'{metric}_med']    = np.percentile(vals, 50)
            record[f'{metric}_q75']    = np.percentile(vals, 75)
            record[f'{metric}_max']    = np.percentile(vals, 100)
        records.append(record)

    return pd.DataFrame(records)

def normalize_val(row, col, metric, norm_values):
    norm = norm_values[metric]
    return row[col] / norm if norm != 0 else row[col]


def find_closest_stage2(stim_box, no_stim_box, metrics=None, normalize=False):
    all_metrics = ['Fbr', 'Fd', 'Ft', 'Fa', 'Fbe', 'Pdobr', 'Ptobr', 'Paobr', 'Pbobr']
    suffixes    = ['_med']#['_min', '_q25', '_med', '_q75', '_max']
    
    if metrics is None:
        metrics = all_metrics

    normalization = {}
    for m in metrics:
        normalization[m] = np.abs(
            np.max(no_stim_box[f'{m}_med']) - np.min(no_stim_box[f'{m}_med'])
        )

    dist = 0
    for m in metrics:
        ref = normalization[m] if normalization[m] != 0 else 1
        
        for s in suffixes:
            col = f'{m}{s}'
            
            val_stim = stim_row[col]
            val_ns   = ns_row[col]
            
            dist += abs(val_stim - val_ns) / ref

    if normalize:
        ref_row     = no_stim_box[(no_stim_box['b'] == 0) & (no_stim_box['t'] == 0)].iloc[0]
        norm_values = {m: ref_row[f'{m}_med'] for m in metrics}
    else:
        norm_values = {m: 1 for m in metrics}

    records = []
    for _, stim_row in stim_box.iterrows():
        best_dist = np.inf
        best_b, best_t = None, None

        for _, ns_row in no_stim_box.iterrows():
            dist = sum(
                np.abs(normalize_val(stim_row, col, m, norm_values) - normalize_val(ns_row, col, m, norm_values))
                for m in metrics for col in [f'{m}{s}' for s in suffixes]
            )
            if dist < best_dist:
                best_dist = dist
                best_b = ns_row['b']
                best_t = ns_row['t']

        records.append({
            'b'       : stim_row['b'],
            't'       : stim_row['t'],
            'sp'      : stim_row['sp'],
            'f'       : stim_row['f'],
            'best_b'  : best_b,
            'best_t'  : best_t,
            'distance': best_dist,
        })

    return pd.DataFrame(records)

def find_all_distances(stim_box, no_stim_box, metrics=None):
    all_metrics = ['Fbr', 'Fd', 'Ft', 'Fa', 'Fbe', 'Pdobr', 'Ptobr', 'Paobr', 'Pbobr']
    suffix = '_med'
    
    if metrics is None:
        metrics = all_metrics

    # NOrmalization with the maximum non stimulated value range
    normalization = {}
    for m in metrics:
        col = f'{m}{suffix}'
        rng = np.mean(no_stim_box[col])
        normalization[m] = rng if rng != 0 else 1.0

    all_pairs = []

    for _, stim_row in stim_box.iterrows():
        for _, ns_row in no_stim_box.iterrows():
            
            current_dist = 0
            for m in metrics:
                col = f'{m}{suffix}'
                current_dist += np.abs(stim_row[col]/normalization[m] - ns_row[col]/normalization[m])
            
            all_pairs.append({
                'b_stim': stim_row['b'],
                't_stim': stim_row['t'],
                'sp': stim_row['sp'],
                'f': stim_row['f'],
                'inte': stim_row['inte'],
                'b_ns': ns_row['b'],
                't_ns': ns_row['t'],
                'distance': current_dist
            })

    return pd.DataFrame(all_pairs)

def find_closest_stage(stim_box, no_stim_box, metrics=None):
    all_metrics = ['Fbr', 'Fd', 'Ft', 'Fa', 'Fbe', 'Pdobr', 'Ptobr', 'Paobr', 'Pbobr']
    suffix = '_med'
    
    if metrics is None:
        metrics = all_metrics

    # 1. Calcolo dei fattori di normalizzazione (Range)
    # Fondamentale per far sì che una metrica in [0, 1] pesi quanto una in [0, 100]
    normalization = {}
    for m in metrics:
        col = f'{m}{suffix}'
        rng = np.abs(np.max(no_stim_box[col]) - np.min(no_stim_box[col]))#np.mean(no_stim_box[col])
        normalization[m] = rng if rng != 0 else 1.0

    records = []

    # 2. Doppio ciclo per confronto incrociato
    # Iteriamo su ogni stato simulato con stimolazione (stim_row)
    for _, stim_row in stim_box.iterrows():
        best_dist = np.inf
        best_b, best_t = None, None

        # Lo confrontiamo con tutti gli stati della progressione naturale (ns_row)
        for _, ns_row in no_stim_box.iterrows():
            
            # Calcolo della distanza di Manhattan pesata
            current_dist = 0
            for m in metrics:
                col = f'{m}{suffix}'
                # Differenza pesata per il range della metrica specifica
                #diff = np.abs(stim_row[col]/normalization[m] - ns_row[col]/normalization[m]) 
                diff = np.abs(stim_row[col]/normalization[m] - ns_row[col]/normalization[m]) 
                current_dist += diff

            # Se questa combinazione (b, t) è la più vicina trovata finora, la salviamo
            if current_dist < best_dist:
                best_dist = current_dist
                best_b = ns_row['b']
                best_t = ns_row['t']

        # Salviamo il risultato per questo specifico stage stimolato
        records.append({
            'b_stim': stim_row['b'],
            't_stim': stim_row['t'],
            'sp': stim_row['sp'],
            'f': stim_row['f'],
            'inte': stim_row['inte'],
            'best_b_ns': best_b,
            'best_t_ns': best_t,
            'min_distance': best_dist
        })

    return pd.DataFrame(records)


def find_closest_stage3(stim_box, no_stim_box, metrics=None):
    suffix = '_med'
    ns_values = []

    normalization = {}
    for m in metrics:
        col = f'{m}{suffix}'
        rng = np.mean(no_stim_box[col])
        normalization[m] = rng if rng != 0 else 1.0
        
    for _, ns_row in no_stim_box.iterrows():
                
        current_value = 0
        for m in metrics:
            col = f'{m}{suffix}'
            val = np.abs(ns_row[col]) / normalization[m]
            current_value += val
            
        abeta = int(ns_row['b'])
        tau = int(ns_row['t'])
        ns_values.append((abeta, tau,current_value))

    
    df_ns_values = pd.DataFrame(ns_values, columns=['abeta', 'tau', 'disease_value'])
    df_ns_values.to_csv(r"..\df_ns_values")
    
    s_values = []
    for _, s_row in stim_box.iterrows():
            
        current_value = 0
        for m in metrics:
            col = f'{m}{suffix}'
            val = np.abs(s_row[col]) / normalization[m]
            current_value += val
            
        abeta = int(s_row['b'])
        tau = int(s_row['t'])
        spot = int(s_row['sp'])
        freq = s_row['f']
        inte = s_row['inte']
        s_values.append((abeta, tau, spot, freq, inte, current_value))

    df_stim_values = pd.DataFrame(s_values, columns=['abeta', 'tau', 'spot', 'freq', 'inte', 'disease_value'])
    df_stim_values.to_csv(r"..\df_stim_values")

    """
    print(len(files))
    for file in files:
    """
        #df= pd.read_pickle(file)#r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Tms_app_results_2\tms_seeds_a3_t3_f0.5_s0")
    """df_2 = best_sf(df)
        #print(best)
        #print(bool_arr[35:55]) 
        filename = os.path.basename(file)
        params = filename.replace("tms_seeds_", "")
        output_name = f"super_th_reg_{params}"
        output_path = os.path.join(base_dir, output_name)
        pd.to_pickle(df_2, output_path)

    """
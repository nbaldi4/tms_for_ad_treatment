import os

import matplotlib.pyplot as plt
import numpy as np
import time
import glob
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import tvb.simulator.lab as tvb
from tvb.simulator.lab import *
from scipy.stats import pearsonr, wasserstein_distance
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from pymoo.core.termination import Termination
from pymoo.core.problem import Problem
from pymoo.indicators.hv import HV
from joblib import Parallel, delayed

from global_variable_creation import get_eeg_monitor, get_conn
from utils import psd, ev, preprocess, simulate
import global_variable_creation as gv
from additional_utils import extract_common_names, macro_fc

    
class EEGMetricProblem(Problem):
    def __init__(self, xl, xu, n_jobs=-1):
        super().__init__(n_var=5, n_obj=3, xl=xl, xu=xu)
        self.n_jobs = n_jobs

    def _evaluate(self, X, out, *args, **kwargs):
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(evaluate_simulation)(row) for row in X
        )
        out["F"] = np.array(results)

class CombinedTermination(Termination):
    """
    Termination based on:
      - either max_gen (stop if generation >= max_gen)
      - or hypervolume do not improve more than min_rel_improv (or min_abs_improv)
        for 'patience' consecutive generations.
    """
    def __init__(self,
                 ref_point,
                 max_gen=100,
                 patience=5,
                 min_rel_improv=1e-4,
                 min_abs_improv=None):
        super().__init__()
        self.ref_point = np.array(ref_point)
        self.max_gen = int(max_gen)
        self.patience = int(patience)
        self.min_rel_improv = float(min_rel_improv) if min_rel_improv is not None else None
        self.min_abs_improv = float(min_abs_improv) if min_abs_improv is not None else None

        self.hv_history = []
        self.no_improve_count = 0

    def _update(self, algorithm):
        gen = algorithm.n_gen
        F = algorithm.pop.get("F")

        try:
            hv_now = HV(ref_point=self.ref_point)(F)
        except Exception:
            # if HV can't be computed (ex. empty F), consider hv_now = -inf
            hv_now = -np.inf

        self.hv_history.append(hv_now)

        if gen >= self.max_gen:
            print(f"[Termination] stop: reached max_gen = {self.max_gen} (gen={gen})")
            return True

        if len(self.hv_history) == 1:
            return False

        prev_best = max(self.hv_history[:-1])
        delta = hv_now - prev_best

        improved = False

        if (self.min_abs_improv is not None) and (delta >= self.min_abs_improv):
            improved = True

        if (self.min_rel_improv is not None) and (abs(prev_best) > 0):
            rel = delta / abs(prev_best)
            if rel >= self.min_rel_improv:
                improved = True

        # update non-improving counter
        if improved:
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        if self.no_improve_count >= self.patience:
            print(f"[Termination] stop: HV improvement below threshold for {self.patience} generations "
                  f"(last delta={delta:.3e}, min_rel={self.min_rel_improv}, min_abs={self.min_abs_improv})")
            return True

        return False

def evaluate_simulation(row, sim_time=20000):
    g, velocity, noise, a, b = row

    ttavg, tavg, teeg, eeg = simulate(sim_time=sim_time, g=g, velocity=velocity, noise=noise, a=a, b=b)
    teeg, eeg = preprocess(teeg, eeg)

    regular_mae, reduced_mae = real_vs_simulated_FC(
        fc_healty=np.load(gv.real_FC_data),
        time_series=eeg,
        return_value=True
    )

    psd_mae = real_vs_simulated_PSD(
        matrix=np.load(gv.real_PSD_data),
        time_series=eeg,
        return_value=True
    )

    return regular_mae, reduced_mae, psd_mae

def real_vs_simulated_PSD(matrix, time_series, n_bin=180, fs=100,
                                   label_real="Experimental", label_sim="Simulated", return_value = False, pass_psd_ev = False):

    # =========================
    # PSD reale
    # =========================
    real_psd_mean = np.mean(matrix, axis=0)

    # Taglio e normalizzazione reale
    real_psd_mean = real_psd_mean[:n_bin].astype(float)

    # Evito divisione per zero
    if np.max(real_psd_mean) == 0:
        real_psd_mean = real_psd_mean + 1e-12
    real_psd_mean /= np.max(real_psd_mean)


    # Frequenze reali (uniformi)
    freqs_uniform = np.linspace(0, 45, n_bin)

    # =========================
    # PSD simulata
    # =========================
    simulated_freqs, simulated_psd_mean = psd(time_series_original=time_series, label='sim_psd', return_psd=True)
    peak_freq = simulated_freqs[np.argmax(simulated_psd_mean)]

    # Taglio frequenze a 45 Hz
    mask = simulated_freqs <= 45
    simulated_freqs = simulated_freqs[mask]
    simulated_psd_mean = simulated_psd_mean[mask]

    # =========================
    # Interpolazione simulata su 180 bin
    # =========================
    simulated_psd_resampled = np.interp(freqs_uniform, simulated_freqs, simulated_psd_mean)

    # Normalizzazione simulata
    if np.max(simulated_psd_resampled) == 0:
        simulated_psd_resampled = simulated_psd_resampled + 1e-12
    simulated_psd_resampled /= np.max(simulated_psd_resampled)

    psd_mae = np.sum(np.abs(simulated_psd_resampled-real_psd_mean))

        # Plot
    plt.figure(figsize=(10,4))
    plt.plot(simulated_freqs, simulated_psd_mean, color='b', linewidth=2, label=label_sim)
    plt.plot(freqs_uniform, real_psd_mean, color='r', linewidth=2, label=label_real)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Normalized PSD")
    plt.title(f"PSD: MAE = {psd_mae:.3f}")
    plt.legend()
    plt.grid()
    if gv.save_plots:
        fig_name = f"sim_vs_real_PSD.png"
        os.makedirs(gv.fig_folder, exist_ok=True)
        plt.savefig(os.path.join(gv.fig_folder, fig_name), dpi=200)
    if gv.show_plots:
        plt.show()
    else:
        plt.close()

    if return_value:
        return psd_mae, peak_freq 

def real_vs_stages_PSD(matrix, stage_psd, stage_freqs, n_bin=180, fs=100,
                                   label_real="Experimental", label_sim="Simulated", return_value = False, pass_psd_ev = False):

    # =========================
    # PSD reale
    # =========================
    if matrix.ndim == 2:
        real_psd_mean = np.mean(matrix, axis=0)
    # Se matrix ha 1 dimensione
    else:
        real_psd_mean = matrix

    # Taglio e normalizzazione reale
    real_psd_mean = real_psd_mean[:n_bin].astype(float)

    # Evito divisione per zero
    if np.max(real_psd_mean) == 0:
        real_psd_mean = real_psd_mean + 1e-12
    real_psd_mean /= np.max(real_psd_mean)


    # Frequenze reali (uniformi)
    freqs_uniform = np.linspace(0, 45, n_bin)

    # Taglio frequenze a 45 Hz
    mask = stage_psd <= 45
    stage_psd = stage_psd[mask]
    stage_psd = stage_psd[mask]

    # =========================
    # Interpolazione simulata su 180 bin
    # =========================
    simulated_psd_resampled = np.interp(freqs_uniform, stage_freqs, stage_psd)

    # Normalizzazione simulata
    if np.max(simulated_psd_resampled) == 0:
        simulated_psd_resampled = simulated_psd_resampled + 1e-12
    simulated_psd_resampled /= np.max(simulated_psd_resampled)
    psd_mae = np.sum(np.abs(simulated_psd_resampled-real_psd_mean))
    

        
    """
        # Normalizzo le PSD a distribuzioni di massa (somma = 1)
    real_weights = real_psd_mean.astype(float)
    sim_weights  = simulated_psd_resampled.astype(float)

    s_real = np.sum(real_weights)
    s_sim  = np.sum(sim_weights)
    if s_real == 0:
        real_weights = np.ones_like(real_weights) / len(real_weights)
    else:
        real_weights = real_weights / s_real

    if s_sim == 0:
        sim_weights = np.ones_like(sim_weights) / len(sim_weights)
    else:
        sim_weights = sim_weights / s_sim

    emd_value = wasserstein_distance(u_values=freqs_uniform,
                                    v_values=freqs_uniform,
                                    u_weights=real_weights,
                                    v_weights=sim_weights)
    """

        # Plot
    plt.figure(figsize=(10,4))
    plt.plot(stage_freqs, stage_psd, color='b', linewidth=2, label=label_sim)
    plt.plot(freqs_uniform, real_psd_mean, color='r', linewidth=2, label=label_real)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Normalized PSD")
    plt.ylim(0, 1.01)
    plt.xlim(0, 45)
    plt.title(f"PSD: MAE = {psd_mae:.3f}")
    plt.legend()
    plt.grid()
    if gv.save_plots:
        fig_name = f"sim_vs_real_PSD.png"
        os.makedirs(gv.fig_folder, exist_ok=True)
        plt.savefig(os.path.join(gv.fig_folder, fig_name), dpi=200)
    if gv.show_plots:
        plt.show()
    else:
        plt.close()

    if return_value:
        return psd_mae

def real_vs_simulated_FC(fc_healty, time_series, return_value=False):
    """
    Compares simulated functional connectivity (from TVB) with real EEG-based FC
    at both full and reduced (macro) resolution levels.

    Displays comparison plots with MAE values.
    """

    # Extract EEG correlation matrix
    tsr_corr = ev(time_series)
    corr_eeg = tsr_corr.array_data[..., 0, 0]
    corr_eeg -= np.eye(len(time_series[0, 0, :, 0]))
    np.fill_diagonal(corr_eeg, 0.0)
    corr_eeg = np.clip(corr_eeg, -0.999999, 0.999999)

    # Align and compute macro-level FC
    sub_fc_healty, com_fc_healty = [], []
    for i in range(len(fc_healty[:, 0, 0])):
        matrix1, matrix2 = to_common_indices(corr_eeg, fc_healty[i, :, :])
        com_fc_healty.append(matrix2)
        sub_fc_healty.append(macro_fc(matrix2))
    corr_eeg = matrix1
    sub_corr_eeg = macro_fc(corr_eeg)
    com_fc_healty = np.array(com_fc_healty)
    com_fc_healty = np.clip(com_fc_healty, -0.999999, 0.999999)

    # --- Compute group-level statistics ---
    normal_corr_stack = np.stack(com_fc_healty, axis=0)
    normal_z_stack = np.arctanh(normal_corr_stack)
    normal_z_mean = np.mean(normal_z_stack, axis=0)
    normal_corr_mean = np.tanh(normal_z_mean)
    normal_mae = np.mean(np.abs(normal_corr_mean - corr_eeg))

    # Normalize matrices to [-1, 1] only for visualization purpose
    corr_eeg = 2 * (corr_eeg - np.min(corr_eeg)) / (np.max(corr_eeg) - np.min(corr_eeg)) - 1
    normal_corr_mean = 2 * (normal_corr_mean - np.min(normal_corr_mean)) / (np.max(normal_corr_mean) - np.min(normal_corr_mean)) - 1

    # Plot comparison (regular scale)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    im1 = ax[0].imshow(corr_eeg, cmap='bwr', vmin=-1, vmax=1)
    ax[0].set_title("Regular size simulated FC")
    im2 = ax[1].imshow(normal_corr_mean, cmap='bwr', vmin=-1, vmax=1)
    ax[1].set_title(f"Regular size real FC: MAE = {normal_mae:.3f}")
    plt.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)

    if gv.save_plots:
        fig_name = "Sim_vs_real_regular_size.png"
        os.makedirs(gv.fig_folder, exist_ok=True)
        plt.savefig(os.path.join(gv.fig_folder, fig_name), dpi=200)

    if gv.show_plots:
        plt.show()
    else:
        plt.close()

    sub_corr_eeg = np.clip(sub_corr_eeg, -0.999999, 0.999999)
    # --- Reduced (macro) resolution ---
    sub_fc_healty = np.array(sub_fc_healty)
    sub_fc_healty = np.clip(sub_fc_healty, -0.999999, 0.999999)
    sub_corr_stack = np.stack(sub_fc_healty, axis=0)
    sub_z_stack = np.arctanh(sub_corr_stack)
    sub_z_mean = np.mean(sub_z_stack, axis=0)
    sub_corr_mean = np.tanh(sub_z_mean)
    reduced_mae = np.mean(np.abs(sub_corr_mean - sub_corr_eeg))


    # Normalize for visualization
    sub_corr_eeg = 2 * (sub_corr_eeg - np.min(sub_corr_eeg)) / (np.max(sub_corr_eeg) - np.min(sub_corr_eeg)) - 1
    sub_corr_mean = 2 * (sub_corr_mean - np.min(sub_corr_mean)) / (np.max(sub_corr_mean) - np.min(sub_corr_mean)) - 1

    # Plot comparison (reduced scale)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    im1 = ax[0].imshow(sub_corr_eeg, cmap='bwr', vmin=-1, vmax=1)
    ax[0].set_title("Reduced size simulated FC")
    im2 = ax[1].imshow(sub_corr_mean, cmap='bwr', vmin=-1, vmax=1)
    ax[1].set_title(f"Reduced size real FC: MAE = {reduced_mae:.3f}")
    plt.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)

    if gv.save_plots:
        fig_name = "Sim_vs_real_reduced_size.png"
        os.makedirs(gv.fig_folder, exist_ok=True)
        plt.savefig(os.path.join(gv.fig_folder, fig_name), dpi=200)

    if gv.save_plots:
        fig_name = f"sim_vs_real_PSD.png"
        os.makedirs(gv.fig_folder, exist_ok=True)
        plt.savefig(os.path.join(gv.fig_folder, fig_name), dpi=200)
    if gv.show_plots:
        plt.show()
    else:
        plt.close()

    if return_value:
        return normal_mae, reduced_mae
    
def real_vs_stages_FC(fc_real, fc_regular_stage, fc_reduced_stage, return_value=False):
    """
    Compares simulated functional connectivity (from TVB) with real EEG-based FC
    at both full and reduced (macro) resolution levels.

    Displays comparison plots with MAE values.
    """

    fc_stage_com_ind, fc_real_com_ind = to_common_indices(fc_regular_stage, fc_real)
    normal_mae = np.mean(np.abs(fc_real_com_ind - fc_stage_com_ind))
    fc_reduced_real = macro_fc(fc_real_com_ind)
    fc_reduced_stage = macro_fc(fc_stage_com_ind)
    fc_reduced_real = np.clip(fc_reduced_real, -0.999999, 0.999999)
    fc_reduced_stage = np.clip(fc_reduced_stage, -0.999999, 0.999999)
    reduced_mae = np.mean(np.abs(fc_reduced_real - fc_reduced_stage))

    # Normalize matrices to [-1, 1] only for visualization purpose
    fc_real_com_ind = 2 * (fc_real_com_ind - np.min(fc_real_com_ind)) / (np.max(fc_real_com_ind) - np.min(fc_real_com_ind)) - 1
    fc_stage_com_ind = 2 * (fc_stage_com_ind - np.min(fc_stage_com_ind)) / (np.max(fc_stage_com_ind) - np.min(fc_stage_com_ind)) - 1

    # Plot comparison (regular scale)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    im1 = ax[0].imshow(fc_stage_com_ind, cmap='bwr', vmin=-1, vmax=1)
    ax[0].set_title("Regular size simulated FC")
    im2 = ax[1].imshow(fc_real_com_ind, cmap='bwr', vmin=-1, vmax=1)
    ax[1].set_title(f"Regular size real FC: MAE = {normal_mae:.3f}")
    plt.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)

    if gv.save_plots:
        fig_name = "Sim_vs_real_regular_size.png"
        os.makedirs(gv.fig_folder, exist_ok=True)
        plt.savefig(os.path.join(gv.fig_folder, fig_name), dpi=200)

    if gv.show_plots:
        plt.show()
    else:
        plt.close()

    # Normalize for visualization
    fc_reduced_real = 2 * (fc_reduced_real - np.min(fc_reduced_real)) / (np.max(fc_reduced_real) - np.min(fc_reduced_real)) - 1
    fc_reduced_stage = 2 * (fc_reduced_stage - np.min(fc_reduced_stage)) / (np.max(fc_reduced_stage) - np.min(fc_reduced_stage)) - 1

    # Plot comparison (reduced scale)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    im1 = ax[0].imshow(fc_reduced_stage, cmap='bwr', vmin=-1, vmax=1)
    ax[0].set_title("Reduced size simulated FC")
    im2 = ax[1].imshow(fc_reduced_real, cmap='bwr', vmin=-1, vmax=1)
    ax[1].set_title(f"Reduced size real FC: MAE = {reduced_mae:.3f}")
    plt.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)

    if gv.save_plots:
        fig_name = "Sim_vs_real_reduced_size.png"
        os.makedirs(gv.fig_folder, exist_ok=True)
        plt.savefig(os.path.join(gv.fig_folder, fig_name), dpi=200)

    if gv.save_plots:
        fig_name = f"sim_vs_real_PSD.png"
        os.makedirs(gv.fig_folder, exist_ok=True)
        plt.savefig(os.path.join(gv.fig_folder, fig_name), dpi=200)
    if gv.show_plots:
        plt.show()
    else:
        plt.close()

    if return_value:
        return normal_mae, reduced_mae

def to_common_indices(tvb_matrix, real_matrix):
    """
    Aligns two functional connectivity matrices (from TVB simulation and real EEG data)
    by selecting only the common EEG channel indices.

    Parameters
    ----------
    tvb_matrix : np.ndarray
        TVB 65x65 functional connectivity matrix.
    real_matrix : np.ndarray
        Real EEG 62x62 functional connectivity matrix.

    Returns
    -------
    tvb_matrix_sub : np.ndarray
        Submatrix of tvb_matrix restricted to common channels.
    real_matrix_sub : np.ndarray
        Submatrix of real_matrix restricted to common channels.
    """
    your_path = 'C:/Users/User/OneDrive - University of Pisa/Desktop/TVB_tutorials/Dati_Healthy/'

    # Channel names for real EEG (62 channels)
    eeg_real_data_names = np.load(your_path + 'names_ch.npy')
    eeg_real_data_names = np.array(eeg_real_data_names)

    # Channel names for Brainstorm EEG (65 channels)
    eeg_brainstorm_65_names = np.array([
        'Fp1', 'Fp2', 'F4', 'F3', 'C3', 'C4', 'P4', 'P3', 'O2', 'O1',
        'F8', 'F7', 'T4', 'T3', 'T6', 'T5', 'Pz', 'Fz', 'IO1', 'IO2',
        'AF9', 'AF10', 'F9', 'F10', 'CB1', 'CB2', 'TP7', 'TP9', 'TP10',
        'TP8', 'Oz', 'Iz', 'PO4', 'PO3', 'CP5', 'CP6', 'CP1', 'CP2',
        'FT9', 'FT10', 'FC2', 'FC1', 'AF3', 'AF4', 'FC6', 'FC5', 'CPz',
        'P1', 'POz', 'P2', 'P6', 'C6', 'P5', 'C1', 'C2', 'C5', 'F2',
        'F6', 'F1', 'AF8', 'F5', 'AF7', 'Fpz', 'FCz', 'Cz', 'T8', 'T7',
        'P8', 'P7'
    ])

    common_names = extract_common_names()

    # Find index positions of common channels in both naming systems
    indices_65 = [np.where(eeg_brainstorm_65_names == name)[0][0] for name in common_names]
    indices_62 = [np.where(eeg_real_data_names == name)[0][0] for name in common_names]

    # Extract the aligned submatrices
    tvb_matrix_sub = tvb_matrix[np.ix_(indices_65, indices_65)]
    real_matrix_sub = real_matrix[np.ix_(indices_62, indices_62)]

    return tvb_matrix_sub, real_matrix_sub

def compute_matrices_diff(m1, m2):
    """
    Computes the difference between two matrices and their correlation coefficient.

    Parameters
    ----------
    m1, m2 : np.ndarray
        Matrices of the same shape.

    Outputs
    -------
    Prints Pearson correlation coefficient and displays difference heatmap.
    """
    diff = m1 - m2
    mask = np.triu(np.ones(m1.shape), k=1).astype(bool)
    r, _ = pearsonr(m1[mask], m2[mask])
    print("Correlation coefficient:", r)

    mae = np.mean(np.abs(diff))
    plt.imshow(diff, cmap='bwr', vmin=-1, vmax=1)
    plt.title(f"Mean absolute difference: {mae:.3f}")
    plt.colorbar()
    plt.show()

def analyze_regular_data(real_data):
    """
    Computes the mean and standard deviation of real EEG FC matrices
    in both correlation and Fisher z-space.
    """
    normal_matrices = np.array(real_data)
    normal_corr_stack = np.stack(normal_matrices, axis=0)
    normal_z_stack = np.arctanh(normal_corr_stack)
    normal_z_mean = np.mean(normal_z_stack, axis=0)
    normal_z_std = np.std(normal_z_stack, axis=0)
    mean_std = np.mean(normal_z_std)

    normal_corr_mean = np.tanh(normal_z_mean)
    normal_corr_mean = 2 * (normal_corr_mean - np.min(normal_corr_mean)) / (np.max(normal_corr_mean) - np.min(normal_corr_mean)) - 1

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    im1 = ax[0].imshow(normal_corr_mean, cmap='bwr', vmin=-1, vmax=1)
    ax[0].set_title("Regular size mean FC")
    im2 = ax[1].imshow(normal_z_std, cmap='viridis')
    ax[1].set_title(f"Regular size FC std (z-space): mean std = {mean_std:.3f}")
    plt.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)

    if gv.save_plots:
        fig_name = "Regular_size_real_data.png"
        os.makedirs(gv.fig_folder, exist_ok=True)
        plt.savefig(os.path.join(gv.fig_folder, fig_name), dpi=200)

    if gv.show_plots:
        plt.show()
    else:
        plt.close()

def analyze_reduced_data(real_data):
    """
    Resizes both standard and real EEG FC matrices and computes the mean
    and standard deviation in both correlation and Fisher z-space.
    """
    sub_matrices = np.zeros((len(real_data), gv.partitioning, gv.partitioning))
    
    for i in range(len(real_data)):
        sub_matrices[i, :, :] = macro_fc(matrix=real_data[i, :, :])
    
    # Shift to z-score to compute mean and standard deviation
    # Stack into a 3D array: shape = (n_subjects, n_regions, n_regions)
    sub_corr_stack = np.stack(sub_matrices, axis=0)
    # --- Fisher z-transform ---
    sub_z_stack = np.arctanh(sub_corr_stack)
    # --- Compute mean and std in z-space ---
    sub_z_mean = np.mean(sub_z_stack, axis=0)
    sub_z_std  = np.std(sub_z_stack, axis=0)
    sub_mean_std = np.mean(sub_z_std)
    # --- Convert mean back to correlation space ---
    sub_corr_mean = np.tanh(sub_z_mean)
    sub_corr_mean = 2 * (sub_corr_mean - np.min(sub_corr_mean)) / (np.max(sub_corr_mean) - np.min(sub_corr_mean)) - 1    

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    im1 = ax[0].imshow(sub_corr_mean, cmap='bwr', vmin=-1, vmax=1)
    ax[0].set_title("Reduced size mean FC")
    im2 = ax[1].imshow(sub_z_std, cmap='viridis')
    ax[1].set_title("Reduced size FC std (z-space): mean std = " + str(sub_mean_std))
    plt.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)

    if gv.save_plots:
        fig_name = "Reduced_size_real_data.png"
        os.makedirs(gv.fig_folder, exist_ok=True)
        plt.savefig(os.path.join(gv.fig_folder, fig_name), dpi=200)

    if gv.show_plots:
        plt.show()
    else:
        plt.close()

def compute_spacing(F):
    """
    Compute minimum distance between pareto front points.
    """
    if len(F) < 2:
        return 0.0

    distances = []
    for i in range(len(F)):
        # minimum distance between i and all other points
        others = np.delete(F, i, axis=0)
        d = np.min(np.linalg.norm(F[i] - others, axis=1))
        distances.append(d)

    distances = np.array(distances)
    mean_d = np.mean(distances)

    return np.sqrt(np.mean((distances - mean_d) ** 2))

def nsga_callback(algorithm):
    F = algorithm.pop.get("F")

    # a ref_point is needed to define the worst possible value in metrics space
    hv = HV(ref_point=gv.ref_PT)(F)
    spacing = compute_spacing(F)
    n_nd = len(F)

    convergence_data = {
        "generation": [],
        "hypervolume": [],
        "spacing": [],
        "n_nd": []
    }

    convergence_data["generation"].append(algorithm.n_gen)
    convergence_data["hypervolume"].append(hv)
    convergence_data["spacing"].append(spacing)
    convergence_data["n_nd"].append(n_nd)

    print(f"[Gen {algorithm.n_gen}] HV={hv:.4f} | Spacing={spacing:.4f} | ND={n_nd}")

def NSGAII(g_range = [0.1, 1.1],
          velocity_range = [1, 300],
          noise_range = [0.1, 5],
          a_range = [0.05, 0.150],
          b_range = [0.025, 0.075],
          pop_size = 8,
          n_gen = 1,
          ref_pt = gv.ref_PT,
          max_to_improve = 5
          ):

    convergence_data = {
        "generation": [],
        "hypervolume": [],
        "spacing": [],
        "n_nd": []
    }

    termination_condition = CombinedTermination(
        ref_point=ref_pt,
        max_gen=n_gen,
        patience=max_to_improve,
        # you choose relative or absolute value
        min_rel_improv=1e-3,
        min_abs_improv=None
    )

    xl = np.array([g_range[0], velocity_range[0], noise_range[0], a_range[0], b_range[0]])
    xu = np.array([g_range[1], velocity_range[1], noise_range[1], a_range[1], b_range[1]])

    problem = EEGMetricProblem(xl=xl, xu=xu, n_jobs=-1)

    algorithm = NSGA2(pop_size=pop_size)

    res = minimize(
        problem=problem,
        algorithm=algorithm,
        seed=1,
        termination=termination_condition,
        callback=nsga_callback,
        verbose=True,
        save_history=True
    )

    pareto = pd.DataFrame(
        np.hstack([res.X, res.F]),
        columns=["g","velocity","noise","a","b","M1","M2","M3"]
    )
    pareto.to_csv("pareto_front_nsga2.csv", index=False)
    pd.DataFrame(convergence_data).to_csv("nsga2_convergence_metrics.csv", index=False)
    plot_convergence(convergence_data)

    print("Pareto front saved in pareto_front_nsga2.csv")
    print("Convergence metrics saved in nsga2_convergence_metrics.csv")
    print("Plots saved: hypervolume_convergence.png, spacing_convergence.png, nondominated_convergence.png")

def plot_convergence(convergence):
    gen = convergence["generation"]

    # ---- Hypervolume ----
    plt.figure(figsize=(8,5))
    plt.plot(gen, convergence["hypervolume"], marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    plt.title("Hypervolume convergence")
    plt.grid(True)

    if gv.save_plots:
        fig_name = "hypervolume_convergence.png"
        os.makedirs(gv.fig_folder, exist_ok=True)
        plt.savefig(os.path.join(gv.fig_folder, fig_name), dpi=200)
    
    if gv.show_plots:
        plt.show()
    else:
        plt.close()


    # ---- Spacing ----
    plt.figure(figsize=(8,5))
    plt.plot(gen, convergence["spacing"], marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Spacing (Diversity)")
    plt.title("Diversity convergence (Spacing)")
    plt.grid(True)

    if gv.save_plots:
        fig_name = "spacing_convergence.png"
        os.makedirs(gv.fig_folder, exist_ok=True)
        plt.savefig(os.path.join(gv.fig_folder, fig_name), dpi=200)
    
    if gv.show_plots:
        plt.show()
    else:
        plt.close()

    # ---- Numero ND ----
    plt.figure(figsize=(8,5))
    plt.plot(gen, convergence["n_nd"], marker="o")
    plt.xlabel("Generation")
    plt.ylabel("# non dominated solutions")
    plt.title("Non dominated solutions")

    if gv.save_plots:
        fig_name = "nondominated_convergence.png"
        os.makedirs(gv.fig_folder, exist_ok=True)
        plt.savefig(os.path.join(gv.fig_folder, fig_name), dpi=200)
    
    if gv.show_plots:
        plt.show()
    else:
        plt.close()

def real_vs_simulated_PSD_single_value(matrix, time_series, index, n_bin=180, fs=100,
                                   label_real="Experimental", label_sim="Simulated", return_value = False):

    # =========================
    # PSD reale
    # =========================
    real_psd_mean = matrix[index, :]

    # Taglio e normalizzazione reale
    real_psd_mean = real_psd_mean[:n_bin].astype(float)

    # Evito divisione per zero
    if np.max(real_psd_mean) == 0:
        real_psd_mean = real_psd_mean + 1e-12
    real_psd_mean /= np.max(real_psd_mean)


    # Frequenze reali (uniformi)
    freqs_uniform = np.linspace(0, 45, n_bin)

    # =========================
    # PSD simulata
    # =========================
    simulated_freqs, simulated_psd_mean = psd(time_series=time_series, label='sim_psd', return_psd=True)
    peak_freq = simulated_freqs[np.argmax(simulated_psd_mean)]

    # Taglio frequenze a 45 Hz
    mask = simulated_freqs <= 45
    simulated_freqs = simulated_freqs[mask]
    simulated_psd_mean = simulated_psd_mean[mask]

    # =========================
    # Interpolazione simulata su 180 bin
    # =========================
    simulated_psd_resampled = np.interp(freqs_uniform, simulated_freqs, simulated_psd_mean)

    # Normalizzazione simulata
    if np.max(simulated_psd_resampled) == 0:
        simulated_psd_resampled = simulated_psd_resampled + 1e-12
    simulated_psd_resampled /= np.max(simulated_psd_resampled)

    psd_mae = np.sum(np.abs(simulated_psd_resampled-real_psd_mean))

        # Normalizzo le PSD a distribuzioni di massa (somma = 1)
    real_weights = real_psd_mean.astype(float)
    sim_weights  = simulated_psd_resampled.astype(float)

    s_real = np.sum(real_weights)
    s_sim  = np.sum(sim_weights)
    if s_real == 0:
        real_weights = np.ones_like(real_weights) / len(real_weights)
    else:
        real_weights = real_weights / s_real

    if s_sim == 0:
        sim_weights = np.ones_like(sim_weights) / len(sim_weights)
    else:
        sim_weights = sim_weights / s_sim

    emd_value = wasserstein_distance(u_values=freqs_uniform,
                                    v_values=freqs_uniform,
                                    u_weights=real_weights,
                                    v_weights=sim_weights)
    
        # Plot
    plt.figure(figsize=(10,4))
    plt.plot(simulated_freqs, simulated_psd_mean, color='b', linewidth=2, label=label_sim)
    plt.plot(freqs_uniform, real_psd_mean, color='r', linewidth=2, label=label_real)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Normalized PSD")
    plt.title(f"PSD: MAE = {psd_mae:.3f}")
    plt.legend()
    plt.grid()
    if gv.save_plots:
        fig_name = f"sim_vs_real_PSD.png"
        os.makedirs(gv.fig_folder, exist_ok=True)
        plt.savefig(os.path.join(gv.fig_folder, fig_name), dpi=200)
    if gv.show_plots:
        plt.show()
    else:
        plt.close()

    if return_value:
        return psd_mae, peak_freq 

def real_vs_simulated_FC_single_value(fc_healty, time_series, index, return_value=False):
    """
    Compares simulated functional connectivity (from TVB) with real EEG-based FC
    at both full and reduced (macro) resolution levels.

    Displays comparison plots with MAE values.
    """

    # Extract EEG correlation matrix
    tsr_corr = ev(time_series)
    corr_eeg = tsr_corr.array_data[..., 0, 0]
    corr_eeg -= np.eye(len(time_series[0, 0, :, 0]))
    np.fill_diagonal(corr_eeg, 0.0)
    corr_eeg = np.clip(corr_eeg, -0.999999, 0.999999)

    # Align and compute macro-level FC
    matrix1, matrix2 = to_common_indices(corr_eeg, fc_healty[index, :, :])
    com_fc_healty = matrix2
    sub_fc_healty = macro_fc(matrix2)
    corr_eeg = matrix1
    sub_corr_eeg = macro_fc(corr_eeg)
    com_fc_healty = np.clip(com_fc_healty, -0.999999, 0.999999)

    # --- Compute group-level statistics ---
    normal_mae = np.mean(np.abs(com_fc_healty - corr_eeg))

    # Normalize matrices to [-1, 1] only for visualization purpose
    corr_eeg = 2 * (corr_eeg - np.min(corr_eeg)) / (np.max(corr_eeg) - np.min(corr_eeg)) - 1
    com_fc_healty = 2 * (com_fc_healty - np.min(com_fc_healty)) / (np.max(com_fc_healty) - np.min(com_fc_healty)) - 1

    # Plot comparison (regular scale)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    im1 = ax[0].imshow(corr_eeg, cmap='bwr', vmin=-1, vmax=1)
    ax[0].set_title("Regular size simulated FC")
    im2 = ax[1].imshow(com_fc_healty, cmap='bwr', vmin=-1, vmax=1)
    ax[1].set_title(f"Regular size real FC: MAE = {normal_mae:.3f}")
    plt.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)

    if gv.save_plots:
        fig_name = "Sim_vs_real_regular_size.png"
        os.makedirs(gv.fig_folder, exist_ok=True)
        plt.savefig(os.path.join(gv.fig_folder, fig_name), dpi=200)

    if gv.show_plots:
        plt.show()
    else:
        plt.close()

    sub_corr_eeg = np.clip(sub_corr_eeg, -0.999999, 0.999999)
    # --- Reduced (macro) resolution ---)
    reduced_mae = np.mean(np.abs(sub_fc_healty - sub_corr_eeg))


    # Normalize for visualization
    sub_corr_eeg = 2 * (sub_corr_eeg - np.min(sub_corr_eeg)) / (np.max(sub_corr_eeg) - np.min(sub_corr_eeg)) - 1
    sub_fc_healty = 2 * (sub_fc_healty - np.min(sub_fc_healty)) / (np.max(sub_fc_healty) - np.min(sub_fc_healty)) - 1

    # Plot comparison (reduced scale)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    im1 = ax[0].imshow(sub_corr_eeg, cmap='bwr', vmin=-1, vmax=1)
    ax[0].set_title("Reduced size simulated FC")
    im2 = ax[1].imshow(sub_fc_healty, cmap='bwr', vmin=-1, vmax=1)
    ax[1].set_title(f"Reduced size real FC: MAE = {reduced_mae:.3f}")
    plt.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)

    if gv.save_plots:
        fig_name = "Sim_vs_real_reduced_size.png"
        os.makedirs(gv.fig_folder, exist_ok=True)
        plt.savefig(os.path.join(gv.fig_folder, fig_name), dpi=200)

    if gv.save_plots:
        fig_name = f"sim_vs_real_PSD.png"
        os.makedirs(gv.fig_folder, exist_ok=True)
        plt.savefig(os.path.join(gv.fig_folder, fig_name), dpi=200)
    if gv.show_plots:
        plt.show()
    else:
        plt.close()

    if return_value:
        return normal_mae, reduced_mae
    
def load_simulation_results(sim_id):
    # ... [il corpo della funzione è lo stesso] ...
    ts_data_dir = os.path.join(gv.data_folder, f"{sim_id}")
    try:
        ttavg = np.load(os.path.join(ts_data_dir, f"ttavg_{sim_id}.npy"))
        tavg = np.load(os.path.join(ts_data_dir, f"tavg_{sim_id}.npy"))
        teeg = np.load(os.path.join(ts_data_dir, f"teeg_{sim_id}.npy"))
        eeg = np.load(os.path.join(ts_data_dir, f"eeg_{sim_id}.npy"))
        print(f"Dati per simulazione '{sim_id}' caricati con successo.")
        return ttavg, tavg, teeg, eeg
    except FileNotFoundError as e:
        print(f"ERRORE: Impossibile trovare i file per sim_id '{sim_id}'.")
        print(f"Percorso cercato: {e.filename}")
        return None, None, None, None

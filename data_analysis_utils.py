import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from utils import simulate, preprocess, functional_connectivity, ev, NSGA_simulate
import global_variable_creation as gv

"""Risultati salvati in 'results_figures/grid_search_results.csv'
Miglior regular triplet: [np.float64(0.27444444444444444), np.float64(18.0), np.float64(34.0)] (MAE=0.2086)
Miglior reduced triplet: [np.float64(0.31), np.float64(18.0), np.float64(37.0)] (MAE=0.1112)"""

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

def extract_common_names():
    """
    Finds the set of EEG channel names common between Brainstorm (65 channels)
    and the real EEG dataset (62 channels).

    Returns
    -------
    common_names : list of str
        Ordered list of channel names present in both datasets.
    """
    your_path = 'C:/Users/User/OneDrive - University of Pisa/Desktop/TVB_tutorials/Dati_Healthy/'
    eeg_real_data_names = np.load(your_path + 'names_ch.npy')
    eeg_real_data_names = np.array(eeg_real_data_names)

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

    eeg_real_set = set(eeg_real_data_names)
    eeg_brainstorm_set = set(eeg_brainstorm_65_names)
    common_names = eeg_real_set.intersection(eeg_brainstorm_set)

    # Keep Brainstorm order
    common_names = [name for name in eeg_brainstorm_65_names if name in eeg_real_set]
    return common_names

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

def match_name(name, coord_names):
    """Helper function to match partial channel names to coordinate entries."""
    for c in coord_names:
        if name in c:
            return c
    raise ValueError(f"{name} not found in coordinate file.")

def quadrants_division(coord_file=r'C:\Users\User\OneDrive - University of Pisa\Desktop\TVB_Distribution\tvb_data\Lib\site-packages\tvb_data\sensors\eeg_brainstorm_65.txt'):
    """
    Divides EEG channels into 4 macroareas (quadrants) based on spatial coordinates.
    """
    common_names = extract_common_names()

    # Read coordinates
    data = []
    with open(coord_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                name, x, y, z = parts
                data.append((name, float(x), float(y), float(z)))

    coord_dict = {name: np.array([x, y, z]) for name, x, y, z in data}
    coord_names = list(coord_dict.keys())

    # Extract coordinates for common channels
    coords_common = np.array([coord_dict[match_name(name, coord_names)] for name in common_names])

    # Compute center for reference
    center = np.mean(coords_common, axis=0)
    cx, cy, cz = center

    # Assign sensors to quadrants
    quadrants = {1: [], 2: [], 3: [], 4: []}
    for i, coord in enumerate(coords_common):
        x, y, z = coord
        if x >= cx and y >= cy:
            quadrants[1].append(i)
        elif x < cx and y >= cy:
            quadrants[2].append(i)
        elif x < cx and y < cy:
            quadrants[3].append(i)
        else:
            quadrants[4].append(i)

    return quadrants

def sestants_division(coord_file=r'C:\Users\User\OneDrive - University of Pisa\Desktop\TVB_Distribution\tvb_data\Lib\site-packages\tvb_data\sensors\eeg_brainstorm_65.txt'):
    """
    Divides EEG channels into 6 macroareas (sestants) based on spatial coordinates.
    """
    common_names = extract_common_names()

    # Read coordinates
    data = []
    with open(coord_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                name, x, y, z = parts
                data.append((name, float(x), float(y), float(z)))

    coord_dict = {name: np.array([x, y, z]) for name, x, y, z in data}
    coord_names = list(coord_dict.keys())
    coords_common = np.array([coord_dict[match_name(name, coord_names)] for name in common_names])

    # Compute coordinate ranges
    width_y = np.abs(np.max(coords_common[:, 1]) - np.min(coords_common[:, 1]))
    cx = np.abs(np.max(coords_common[:, 0]) + np.min(coords_common[:, 0])) / 2

    sestants = {i: [] for i in range(1, 7)}
    for i, coord in enumerate(coords_common):
        x, y, z = coord
        if x >= cx and y >= (width_y * 0.67 + np.min(coords_common[:, 1])):
            sestants[1].append(i)
        elif x >= cx and (width_y/3 + np.min(coords_common[:, 1])) <= y < (width_y * 0.67 + np.min(coords_common[:, 1])):
            sestants[2].append(i)
        elif x >= cx and y < (width_y/3 + np.min(coords_common[:, 1])):
            sestants[3].append(i)
        elif x < cx and y >= (width_y * 0.67 + np.min(coords_common[:, 1])):
            sestants[4].append(i)
        elif x < cx and (width_y/3 + np.min(coords_common[:, 1])) <= y < (width_y * 0.67 + np.min(coords_common[:, 1])):
            sestants[5].append(i)
        else:
            sestants[6].append(i)
    return sestants

def macro_fc(matrix):
    """
    Computes a macro-scale functional connectivity matrix by averaging
    FC values over predefined brain regions (quadrants or sestants).

    Returns
    -------
    macro_fc : np.ndarray
        Aggregated matrix of shape (gv.partitioning, gv.partitioning)
    """
    if gv.partitioning == 4:
        macroareas = quadrants_division()
    elif gv.partitioning == 6:
        macroareas = sestants_division()
    else:
        raise ValueError("Unsupported number of macroareas.")

    macro_fc = np.zeros((gv.partitioning, gv.partitioning))
    for i in range(1, gv.partitioning + 1):
        for j in range(1, gv.partitioning + 1):
            idx_i = macroareas[i]
            idx_j = macroareas[j]
            submatrix = matrix[np.ix_(idx_i, idx_j)]
            macro_fc[i - 1, j - 1] = np.mean(submatrix)
    return macro_fc

def simulation_vs_real_data_FC(fc_healty):
    """
    Compares simulated functional connectivity (from TVB) with real EEG-based FC
    at both full and reduced (macro) resolution levels.

    Displays comparison plots with MAE values.
    """
    # Run TVB simulation
    ttavg, tavg, teeg, eeg = simulate(sim_time=20000, cip=0, cep=0, taue=0)
    teeg, eeg = preprocess(teeg, eeg)

    # Extract EEG correlation matrix
    tsr_corr = ev(eeg)
    corr_eeg = tsr_corr.array_data[..., 0, 0]
    corr_eeg -= np.eye(len(eeg[0, 0, :, 0]))
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

    plt.show()

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

    plt.show()

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

    plt.show()


def NSGA_simulation_vs_real_data_FC(fc_healty, sim_time, g, velocity, noise, a, b):
    """
    Compares simulated functional connectivity (from TVB) with real EEG-based FC
    at both full and reduced (macro) resolution levels.

    Displays comparison plots with MAE values.
    """
    # Run TVB simulation
    ttavg, tavg, teeg, eeg = NSGA_simulate(sim_time=sim_time, cip=0, cep=0, g=g, velocity=velocity, noise=noise, a=a, b=b)
    teeg, eeg = preprocess(teeg, eeg)

    # Extract EEG correlation matrix
    tsr_corr = ev(eeg)
    corr_eeg = tsr_corr.array_data[..., 0, 0]
    corr_eeg -= np.eye(len(eeg[0, 0, :, 0]))
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

    sub_corr_eeg = np.clip(sub_corr_eeg, -0.999999, 0.999999)
    # --- Reduced (macro) resolution ---
    sub_fc_healty = np.array(sub_fc_healty)
    sub_fc_healty = np.clip(sub_fc_healty, -0.999999, 0.999999)
    sub_corr_stack = np.stack(sub_fc_healty, axis=0)
    sub_z_stack = np.arctanh(sub_corr_stack)
    sub_z_mean = np.mean(sub_z_stack, axis=0)
    sub_corr_mean = np.tanh(sub_z_mean)
    reduced_mae = np.mean(np.abs(sub_corr_mean - sub_corr_eeg))

    return normal_mae, reduced_mae

"""
def healty_parameters_grid_search(impaired_regions, struct_conn, conn, fc_healty, partitioning, tau_e_values, C_ip_values, C_ep_values):

    # === Crea la cartella di output se non esiste ===
    output_dir = "results_figures"
    os.makedirs(output_dir, exist_ok=True)

    results = []  # ogni elemento sarà [tau_e, C_e, C_i, regular_mae, reduced_mae]
    smallest_regular_mae = 52*52
    smallest_reduced_mae = 6*6
    for i in range(len(tau_e_values)):
        for j in range(len(C_ep_values)):
            for k in range(len(C_ip_values)):
                ttavg, tavg, teeg, eeg = Simulate_Niccolò(impaired_regions = impaired_regions, 
                                                          structural_connectivities = struct_conn, 
                                                          cp = 16, lp = 28, np_parameter = 3,
                                                          g=1, velocity=np.inf, noise=50e-3, sim_time = 20000, 
                                                          taue=i, cep=j, cip=k, tau_e_values=tau_e_values, 
                                                          C_ip_values=C_ip_values, C_ep_values=C_ep_values)
                ttavg, tavg, teeg, eeg = preprocess(ttavg, tavg, teeg, eeg)
                corr_eeg, corr_eeg_th = functional_connectivity(eeg, conn)
                com_fc_healty = []
                sub_fc_healty = []
                for l in range(len(fc_healty[:,0,0])):
                    matrix1, matrix2 = to_common_indices(corr_eeg, fc_healty[l,:,:])
                    com_fc_healty.append(matrix2)
                    sub_fc_healty.append(macro_fc(matrix2, partitioning))
                corr_eeg = matrix1

                sub_corr_eeg = macro_fc(corr_eeg, partitioning)
                sub_corr_eeg /= (np.max(sub_corr_eeg) - np.min(sub_corr_eeg))
                sub_corr_eeg -= np.mean(sub_corr_eeg)
                corr_eeg /= (np.max(corr_eeg) - np.min(corr_eeg))
                corr_eeg -= np.mean(corr_eeg)
                com_fc_healty = np.array(com_fc_healty)
                sub_fc_healty = np.array(sub_fc_healty)
                # Per calcolare media e deviazione standard passo allo z-score
                # Stack into a 3D array: shape = (n_subjects, n_regions, n_regions)
                normal_corr_stack = np.stack(com_fc_healty, axis=0)
                    # --- Fisher z-transform ---
                normal_z_stack = np.arctanh(normal_corr_stack)

                    # --- Compute mean and std in z-space ---
                normal_z_mean = np.mean(normal_z_stack, axis=0)

                    # --- Convert mean back to correlation space ---
                normal_corr_mean = np.tanh(normal_z_mean)
                
                #normal_corr_mean /= (np.max(normal_corr_mean) - np.min(normal_corr_mean))
                #normal_corr_mean -= np.mean(normal_corr_mean)
                normal_mae = np.mean(np.abs(normal_corr_mean - corr_eeg))

                if normal_mae < smallest_regular_mae:
                    smallest_regular_mae = normal_mae
                    best_regular_triplet = [tau_e_values[i], C_ep_values[j], C_ip_values[k]]

                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                im1 = ax[0].imshow(corr_eeg, cmap='bwr', vmin=-1, vmax=1)
                ax[0].set_title("Regular size simulated FC")
                im2 = ax[1].imshow(normal_corr_mean, cmap='bwr', vmin=-1, vmax=1)
                ax[1].set_title("Regular size real FC: mae = " + str(normal_mae))
                plt.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
                plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)
                plt.tight_layout()
                
                # === Salva la figura ===
                fig_name = f"FC_full__tau_e{tau_e_values[i]}__C_e{C_ep_values[j]}__C_i{C_ip_values[k]}.png"
                fig_path = os.path.join(output_dir, fig_name)
                plt.savefig(fig_path, dpi=200)
                plt.close(fig)

                    # Per calcolare media e deviazione standard passo allo z-score
                # Stack into a 3D array: shape = (n_subjects, n_regions, n_regions)
                sub_corr_stack = np.stack(sub_fc_healty, axis=0)
                    # --- Fisher z-transform ---
                sub_z_stack = np.arctanh(sub_corr_stack)

                    # --- Compute mean and std in z-space ---
                sub_z_mean = np.mean(sub_z_stack, axis=0)

                    # --- Convert mean back to correlation space ---
                sub_corr_mean = np.tanh(sub_z_mean)
                
                #sub_corr_mean= analyze_reduced_data(real_data, partitioning, visualizing = False)
                #sub_corr_mean = macro_fc(normal_corr_mean, partitioning)
                #sub_corr_mean /= (np.max(sub_corr_mean) - np.min(sub_corr_mean))
                #sub_corr_mean -= np.mean(sub_corr_mean)
                reduced_mae = np.mean(np.abs(sub_corr_mean - sub_corr_eeg))

                
                if reduced_mae < smallest_reduced_mae:
                    smallest_reduced_mae = reduced_mae
                    best_reduced_triplet = [tau_e_values[i], C_ep_values[j], C_ip_values[k]]
                
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                im1 = ax[0].imshow(sub_corr_eeg, cmap='bwr', vmin=-1, vmax=1)
                ax[0].set_title("Reduced size simulated FC")
                im2 = ax[1].imshow(sub_corr_mean, cmap='bwr', vmin=-1, vmax=1)
                ax[1].set_title("Reduced size real FC: mae = " + str(reduced_mae))
                plt.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
                plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)
                plt.tight_layout()

                # === Salva la figura ===
                fig_name = f"FC_reduced__tau_e{tau_e_values[i]}__C_e{C_ep_values[j]}__C_i{C_ip_values[k]}.png"
                fig_path = os.path.join(output_dir, fig_name)
                plt.savefig(fig_path, dpi=200)
                plt.close(fig)

                # --- Salva nel vettore risultati ---
                results.append([
                    tau_e_values[i],
                    C_ep_values[j],
                    C_ip_values[k],
                    normal_mae,
                    reduced_mae
                ])

    results = np.array(results)
    np.savetxt(os.path.join(output_dir, "grid_search_results.csv"),
               results,
               delimiter=",",
               header="tau_e,C_e,C_i,regular_mae,reduced_mae",
               comments='')
    
    # NPY (più compatto e veloce da ricaricare in Python)
    np.save(os.path.join(output_dir, "grid_search_results.npy"), results)

    print("Risultati salvati in 'results_figures/grid_search_results.csv'")
    print(f"Miglior regular triplet: {best_regular_triplet} (MAE={smallest_regular_mae:.4f})")
    print(f"Miglior reduced triplet: {best_reduced_triplet} (MAE={smallest_reduced_mae:.4f})")


your_path = 'C:/Users/User/OneDrive - University of Pisa/Desktop/TVB_tutorials/'
struct_conn = np.load(your_path + 'structural_connectivities.npy')
impaired_regions = [21,22,30,31,32,34,59,60,68,69,70,72]
conn =  connectivity.Connectivity.from_file('connectivity_76.zip')
your_path = 'C:/Users/User/OneDrive - University of Pisa/Desktop/TVB_tutorials/Dati_Healthy/'
fc_healty = np.load(your_path + 'fc_preview_ctr_alpha.npy')
healty_parameters_grid_search(impaired_regions, struct_conn, conn, fc_healty, partitioning=6, tau_e_values = np.linspace(0.270,0.310,10), C_ip_values = np.linspace(37,10,10), C_ep_values = np.linspace(110,18,20))




ttavg, tavg, teeg, eeg = Simulate_Niccolò(impaired_regions = impaired_regions, structural_connectivities = struct_conn, cp = 16, lp = 28, np_parameter = 3,
                                  g=1, velocity=np.inf, noise=50e-3, sim_time = 20000)

your_path = 'C:/Users/User/OneDrive - University of Pisa/Desktop/TVB_tutorials/'
struct_conn = np.load(your_path + 'structural_connectivities.npy')
impaired_regions = [21,22,30,31,32,34,59,60,68,69,70,72]
ttavg, tavg, teeg, eeg = Simulate(impaired_regions = impaired_regions, structural_connectivities = struct_conn, cp = 0, lp = 0, np_parameter = 3,
                                  g=1, velocity=np.inf, noise=50e-3, sim_time = 20)
#ttavg, tavg, teeg, eeg = preprocess(ttavg, tavg, teeg, eeg)
tavg /= (np.max(tavg,0) - np.min(tavg,0 ))
eeg /= (np.max(eeg,0) - np.min(eeg,0 ))

corr, corr_th = functional_connectivity(eeg, conn=conn)
your_path = 'C:/Users/User/OneDrive - University of Pisa/Desktop/TVB_tutorials/Dati_Healthy/'
fc_healty = np.load(your_path + 'fc_preview_ctr_alpha.npy')
#A, B = to_common_indices(corr, fc_healty[1,:,:])
simulation_vs_real_data_FC(impaired_regions, struct_conn, conn, fc_healty, 6)
"""
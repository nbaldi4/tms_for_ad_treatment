import numpy as np
import os

import global_variable_creation as gv

def extract_common_names():
    """
    Finds the set of EEG channel names common between Brainstorm (65 channels)
    and the real EEG dataset (62 channels).

    Returns
    -------
    common_names : list of str
        Ordered list of channel names present in both datasets.
    """
    your_path = r"..\HC_data"
    eeg_real_data_names = np.load(os.path.join(your_path, 'names_ch.npy'))
    eeg_real_data_names = np.array(eeg_real_data_names)
    monitor_eeg = gv.get_eeg_monitor()
    sensor_names = np.array(monitor_eeg.sensors.labels)
    """
    sensor_names = np.array(['Cz' 'POz' 'P3' 'Pz' 'FCz' 'FC5' 'P1' 'P6' 'O2' 'O1' 'Fpz' 'P4' 'TP7'
        'PO8' 'FT7' 'FT8' 'Fz' 'CPz' 'CP3' 'AF8' 'PO7' 'C3' 'C2' 'C1' 'AF7' 'C6'
        'C5' 'C4' 'FC6' 'FC1' 'FC2' 'FC3' 'FC4' 'P7' 'T8' 'P5' 'T7' 'P8' 'P9'
        'PO3' 'PO4' 'AF4' 'F3' 'Fp1' 'Oz' 'Fp2' 'F1' 'F2' 'Iz' 'F4' 'F5' 'F6'
        'F7' 'F8' 'P10' 'CP5' 'P2' 'AFz' 'CP1' 'CP2' 'TP8' 'CP4' 'AF3' 'CP6'
        ])
    """
    """
    eeg_brainstorm_65_names = np.array(['Cz', 'POz', 'P3', 'Pz', 'FCz', 'FC5', 'P1', 'P6', 'O2', 'O1', 'Fpz', 'P4', 'TP7',
        'PO8', 'FT7', 'FT8', 'Fz', 'CPz', 'CP3', 'AF8', 'PO7', 'C3', 'C2', 'C1', 'AF7', 'C6',
        'C5', 'C4', 'FC6','FC1', 'FC2', 'FC3', 'FC4', 'P7', 'T8', 'P5', 'T7', 'P8', 'P9',
        'PO3', 'PO4', 'AF4', 'F3', 'Fp1', 'Oz', 'Fp2' ,'F1' ,'F2' ,'Iz' ,'F4' ,'F5', 'F6',
        'F7' ,'F8' ,'P10', 'CP5', 'P2', 'AFz', 'CP1', 'CP2', 'TP8', 'CP4' ,'AF3', 'CP6'
        ]) #from TVB eeg_brainstorm_65.txt
    """
    eeg_real_set = set(eeg_real_data_names)
    eeg_sim_set = set(sensor_names)
    common_names = eeg_real_set.intersection(eeg_sim_set)

    # Keep Brainstorm order
    common_names = [name for name in sensor_names if name in eeg_real_set]
    #print(common_names)
    return common_names


def match_name(name, coord_names):
    """Helper function to match partial channel names to coordinate entries."""
    for c in coord_names:
        if name in c:
            return c
    raise ValueError(f"{name} not found in coordinate file.")

def quadrants_division(coord_file=r"..\HC_data\sub-CLB00044_EEG_Locations.txt"):
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

def sestants_division(coord_file=r"..\HC_data\sub-CLB00044_EEG_Locations.txt"):
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
    cx = (np.max(coords_common[:, 0]) + np.min(coords_common[:, 0])) / 2

    sestants = {i: [] for i in range(1, 7)}
    """
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
    """
    for i, coord in enumerate(coords_common):
        x, y, z = coord
        if x < cx and y >= (width_y * 0.67 + np.min(coords_common[:, 1])):
            sestants[1].append(i)
        elif x < cx and (width_y/3 + np.min(coords_common[:, 1])) <= y < (width_y * 0.67 + np.min(coords_common[:, 1])):
            sestants[2].append(i)
        elif x < cx and y < (width_y/3 + np.min(coords_common[:, 1])):
            sestants[3].append(i)
        elif x >= cx and y >= (width_y * 0.67 + np.min(coords_common[:, 1])):
            sestants[4].append(i)
        elif x >= cx and (width_y/3 + np.min(coords_common[:, 1])) <= y < (width_y * 0.67 + np.min(coords_common[:, 1])):
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
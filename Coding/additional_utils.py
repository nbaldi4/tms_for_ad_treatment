import numpy as np

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
        'F6', 'F1', 'AF8', 'F5', 'AF7', 'Fpz', 'FCz', 'Cz'
    ]) #from TVB eeg_brainstorm_65.txt

    eeg_real_set = set(eeg_real_data_names)
    eeg_brainstorm_set = set(eeg_brainstorm_65_names)
    common_names = eeg_real_set.intersection(eeg_brainstorm_set)

    # Keep Brainstorm order
    common_names = [name for name in eeg_brainstorm_65_names if name in eeg_real_set]
    return common_names

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
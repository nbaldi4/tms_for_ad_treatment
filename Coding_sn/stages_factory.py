#This file only purpose is to one time create a matrix for each parameter (a and b of JR model). Each of this matrix has 4 raws corresponding sto HC, 
# and AD stages 1, 2 and 3 and 76 columns corresponding to the 76 connectivity areas.

import numpy as np
import math

from global_variable_creation import get_conn
import global_variable_creation as gv


def compute_stage_matrix(matrix, stage1 = [], stage2= [], stage3= [], 
                         deg_perc = [0, (1/(1+19**(1/3)))/2, (1/(1+19**(-1/3)))/2, 0.95/2], #[0, 0.14, 0.39, 0.475],#[0, 0.155, 0.315, 0.45], 
                         w_ij_delta = 0.3, w_ii_delta = 0.3, lonely = 3/4):#(1-0.0134/0.0191), w_ii_delta = (1-0.0127/0.0181), lonely = 1/1.33):
    matrix2 = matrix.copy()
    stages = np.concatenate((stage1, stage2, stage3)).astype(int)
    for i in stages:
        for j in range(matrix.shape[0]):
            if (i in stage1) and (j not in stages):
                matrix[i,j] = matrix2[i,j]*(1-(deg_perc[1] + deg_perc[1]) * w_ij_delta * lonely)
                matrix[j,i] = matrix2[j,i]*(1-(deg_perc[1] + deg_perc[1]) * w_ij_delta * lonely)
            if (i in stage2) and (j not in stages):
                matrix[i,j] = matrix2[i,j]*(1-(deg_perc[2] + deg_perc[2]) * w_ij_delta * lonely)
                matrix[j,i] = matrix2[j,i]*(1-(deg_perc[2] + deg_perc[2]) * w_ij_delta * lonely)
            if (i in stage3) and (j not in stages):
                matrix[i,j] = matrix2[i,j]*(1-(deg_perc[3] + deg_perc[3]) * w_ij_delta * lonely)
                matrix[j,i] = matrix2[j,i]*(1-(deg_perc[3] + deg_perc[3]) * w_ij_delta * lonely)
            if (i in stage1) and (j in stage1):
                if i == j:
                    matrix[i,j] = matrix2[i,j]*(1-(deg_perc[1] + deg_perc[1]) * w_ii_delta)
                if i != j:
                    matrix[i,j] = matrix2[i,j]*(1-(deg_perc[1] + deg_perc[1]) * w_ij_delta)
                    matrix[j,i] = matrix2[j,i]*(1-(deg_perc[1] + deg_perc[1]) * w_ij_delta)
            if (i in stage2) and (j in stage2):
                if i == j:
                    matrix[i,j] = matrix2[i,j]*(1-(deg_perc[2] + deg_perc[2]) * w_ii_delta)
                if i != j:
                    matrix[i,j] = matrix2[i,j]*(1-(deg_perc[2] + deg_perc[2]) * w_ij_delta)
                    matrix[j,i] = matrix2[j,i]*(1-(deg_perc[2] + deg_perc[2]) * w_ij_delta)
            if (i in stage3) and (j in stage3):
                if i == j:
                    matrix[i,j] = matrix2[i,j]*(1-(deg_perc[3] + deg_perc[3]) * w_ii_delta)
                if i != j:
                    matrix[i,j] = matrix2[i,j]*(1-(deg_perc[3] + deg_perc[3]) * w_ij_delta)
                    matrix[j,i] = matrix2[j,i]*(1-(deg_perc[3] + deg_perc[3]) * w_ij_delta)
            if (i in stage1) and (j in stage2):
                matrix[i,j] = matrix2[i,j]*(1-(deg_perc[1] + deg_perc[2]) * w_ij_delta)
                matrix[j,i] = matrix2[j,i]*(1-(deg_perc[1] + deg_perc[2]) * w_ij_delta)
            if (i in stage1) and (j in stage3):
                matrix[i,j] = matrix2[i,j]*(1-(deg_perc[1] + deg_perc[3]) * w_ij_delta)
                matrix[j,i] = matrix2[j,i]*(1-(deg_perc[1] + deg_perc[3]) * w_ij_delta)
            if (i in stage2) and (j in stage3):
                matrix[i,j] = matrix2[i,j]*(1-(deg_perc[2] + deg_perc[3]) * w_ij_delta)
                matrix[j,i] = matrix2[j,i]*(1-(deg_perc[2] + deg_perc[3]) * w_ij_delta)
    return matrix

tau_stage_1 = [ # 4 regions
    5,  54,  # EC   (enthorinal cortex)
    15, 64   # PHIG (parahippocampal cortex)
]

tau_stage_2 = [ # 30 regions
    2,  51,  # CACG (caudal anterior cingulate)
    7,  56,  # IPG  (inferior parietal cortex)
    8,  57,  # ITG  (inferior temporal gyrus)
    9,  58,  # ICG  (isthmus cingulate) 
    11, 60,  # LOFG (lateral orbitofrontal)
    13, 62,  # MOFG (medial orbitofrontal)
    14, 63,  # MTG  (middle temporal gyrus)
    24, 73,  # PCU  (precuneus)
    25, 74,  # RACG (rostral anterior cingulate)
    26, 75,  # RMFG (rostral middle frontal gyrus)
    27, 76,  # SFG  (superior frontal gyrus)
    28, 77,  # SPG  (superior parietal cortex)
    29, 78,  # STG  (superior temporal gyrus)
    40, 47,  # HI   (hippocampus)
    41, 48   # AM   (amygdala)
]

tau_stage_3 = [ # 48 regions
    1,  50,  # BSTS (banks of superior temporal sulcus)
    3,  52,  # CMFG (caudal middle frontal gyrus)
    4,  53,  # CU   (cuneus)
    6,  55,  # FG   (fusiform gyrus)
    10, 59,  # LOG  (lateral occipital)
    12, 61,  # LG   (lingual gyrus)
    16, 65,  # PaCG (paracentral lobule)
    17, 66,  # POP  (pars opercularis)
    18, 67,  # POR  (pars orbitalis)
    19, 68,  # PTR  (pars triangularis)
    20, 69,  # PCAL (pericalcarine / primary visual cortex)
    21, 70,  # PoCG (postcentral gyrus / primary somatosensory cortex)
    23, 72,  # PrCG (precentral gyrus / primary motor cortex)
    30, 79,  # SMG  (supramarginal gyrus)
    31, 80,  # FP   (frontal temporal)
    32, 81,  # TP   (temporal pole)
    33, 82,  # TTG  (transverse temporal gyrus / primary auditory cortex)
    34, 83,  # IN   (insula)
    35, 84,  # CER  (cerebellum cortex)
    36, 43,  # TH   (thalamus)
    37, 44,  # CA   (caudate)
    38, 45,  # PU   (putamen)
    39, 46,  # PA   (pallidum)
    42, 49,  # AC   (accumbens) 
]

tau_stage_1 = np.array(tau_stage_1).astype(int)
tau_stage_2 = np.array(tau_stage_2).astype(int)
tau_stage_3 = np.array(tau_stage_3).astype(int)
#tau_stage_all = np.array(tau_stage_all).astype(int)

matrix = get_conn()
tau_matrix_0 = matrix.weights.copy()
tau_matrix_1 = matrix.weights.copy()
tau_matrix_2 = matrix.weights.copy()
tau_matrix_3 = matrix.weights.copy()
tau_matrix_4 = matrix.weights.copy()
tau_matrix_5 = matrix.weights.copy()

tau_matrix_5 = compute_stage_matrix(
    tau_matrix_5, 
    stage3 = np.concatenate((tau_stage_1, tau_stage_2, tau_stage_3)).astype(int),
    )
tau_matrix_4 = compute_stage_matrix(
    tau_matrix_4, 
    stage3 = np.concatenate((tau_stage_1, tau_stage_2)).astype(int),
    stage2 = tau_stage_3
    )
tau_matrix_3 = compute_stage_matrix(
    tau_matrix_3, 
    stage3 = tau_stage_1, 
    stage2 = tau_stage_2, 
    stage1 = tau_stage_3
    )
tau_matrix_2 = compute_stage_matrix(
    tau_matrix_2, 
    stage2 = tau_stage_1, 
    stage1 = tau_stage_2
    )
tau_matrix_1 = compute_stage_matrix(
    tau_matrix_1, 
    stage1 = tau_stage_1
    )


tau_matrices = np.array([tau_matrix_0, tau_matrix_1, tau_matrix_2, tau_matrix_3, tau_matrix_4, tau_matrix_5])
#b_stages = np.array([stage_0_b, stage_1_b, stage_2_b, stage_3_b, stage_4_b, stage_5_b])
#a_stages = np.array([stage_0_a, stage_1_a, stage_2_a, stage_3_a, stage_4_a, stage_5_a])
conteggio = (tau_matrix_0 != 0).sum()
print(np.sum(tau_matrix_0))
print(conteggio)
np.set_printoptions(precision=3, suppress=True)
print(tau_matrix_0[12:16, 12:16])
print(tau_matrix_1[12:16, 12:16])
print(tau_matrix_2[12:16, 12:16])
print(tau_matrix_3[12:16, 12:16])
print(tau_matrix_4[12:16, 12:16])
print(tau_matrix_5[12:16, 12:16])

diagonale = np.diag(tau_matrix_0)

# Conta gli elementi non nulli in quell'array
conteggio_diag = np.count_nonzero(diagonale)
print(np.sum(diagonale))
print(f"Elementi non zero sulla diagonale: {conteggio_diag}")

matrix_ones = np.ones((4,4))
matrix_ones_iniziale = matrix_ones.copy()
#print(matrix_ones)
#print(matrix_ones_iniziale)
matrix_ones_iniziale = compute_stage_matrix(matrix = matrix_ones_iniziale, stage1 = [1], stage2= [2], stage3= [3], 
                         deg_perc = [0, 0.14, 0.39, 0.475])

#print(matrix_ones)
#print(matrix_ones_iniziale)

#np.save(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Degeneration_data\a_stages.npy", a_stages)
#np.save(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Degeneration_data\b_stages.npy", b_stages)
np.save(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Degeneration_data\tau_matrices_connectivity_86.npy", tau_matrices)
#This file only purpose is to one time create a matrix for each parameter (a and b of JR model). Each of this matrix has 4 raws corresponding sto HC, 
# and AD stages 1, 2 and 3 and 76 columns corresponding to the 76 connectivity areas.

import numpy as np
import math

from global_variable_creation import get_conn
import global_variable_creation as gv
"""
#alternative staging
Abeta_stage_0 = []

Abeta_stage_1_b = [
    22, 60,   # PFCPOL (prefrontal polar)
    21, 59,   # PFCORB (orbitofrontal)
    31, 69,   # TCI (inferior temporal)
    30, 68,   # TCC (central temporal)
    34, 72,   # TCV (ventral temporal)
    32, 70    # TCPOL (temporal pole)
]

Abeta_stage_2_b = [
    9, 47,      # Hippocampus (HC)
    24, 62,     # Parahippocampal gyrus (PHC)
    2, 40,      # Amygdala (AMYG)
    10, 48,     # Insula (IA) = anterior + posterior
    20, 58,     # Subgenual cingulum (PFCM)
    5, 43,      # Retrosplenial cingulum (CCR)
    23, 61      # Ventrolateral prefrontal cortex (PFCVL)
]

Abeta_stage_3_b = [
    0,  1,  3,  4,  6,  7,  8, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    25, 26, 27, 28, 29, 33, 35, 36, 37,
    38, 39, 41, 42, 44, 45, 46, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    63, 64, 65, 66, 67, 71, 73, 74, 75
]

tau_stage_0_b = np.array([])

tau_stage_1_b = np.array([
    24, 62   # PHC
])

tau_stage_2_b = np.array([
    9, 47,       # Temporal allocortex / hippocampal limbic III -> HC
    31, 69,      # TCI
    30, 68,      # TCC
    34, 72       # TCV
])

tau_stage_3_b = np.array([
    18, 56,   # PFCDL
    17, 55,   # PFCCL
    19, 57,   # PFCDM
    12, 50,   # M1
    28, 66,   # S1
    29, 67,   # S2
    35, 73,   # V1 
    36, 74    # V2
])


Abeta_stage_0 = []

Abeta_stage_1_b = [
    22, 60,   # PFCPOL (prefrontal polar)
    21, 59,   # PFCORB (orbitofrontal)
    31, 69,   # TCI (inferior temporal)
    30, 68,   # TCC (central temporal)
    34, 72,   # TCV (ventral temporal)
    32, 70    # TCPOL (temporal pole)
]

Abeta_stage_2_b = [
    9, 47,      # Hippocampus (HC)
    24, 62,     # Parahippocampal gyrus (PHC)
    2, 40,      # Amygdala (AMYG)
    10, 48,     # Insula (IA) = anterior + posterior
    20, 58,     # Subgenual cingulum (PFCM)
    5, 43,      # Retrosplenial cingulum (CCR)
    23, 61      # Ventrolateral prefrontal cortex (PFCVL)
]

Abeta_stage_3_b = [
    0,  1,  3,  4,  6,  7,  8, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    25, 26, 27, 28, 29, 33, 35, 36, 37,
    38, 39, 41, 42, 44, 45, 46, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    63, 64, 65, 66, 67, 71, 73, 74, 75
]

Abeta_stage_1 = [ #16 areas
    21, 59,   # PFCORB (orbitofrontal)
    22, 60,   # PFCPOL (prefrontal polar)
    23, 61,   # PFCVL (prefrontal ventrolateral)
    30, 68,   # TCC (central temporal)
    31, 69,   # TCI (inferior temporal)
    32, 70,   # TCPOL (temporal pole)
    33, 71,   # TCS 
    34, 72    # TCV (ventral temporal)
]

Abeta_stage_2 = [ #22 areas
    2, 40,      # Amygdala (AMYG)
    3, 41,      # CCA
    4, 42,      # CCP
    5, 43,      # Retrosplenial cingulum (CCR)
    6, 44,      # Subgenual cingulum (CCS)
    9, 47,      # Hippocampus (HC)
    10, 48,     # Insula (IA) anterior
    11, 49,     # Insula (IA) posterior
    20, 58,     # Subgenual cingulum (PFCM)
    24, 62,     # Parahippocampal gyrus (PHC)
    37, 75      # CC
]

Abeta_stage_3 = [ #34 areas
    0, 1, 12, 13, 14, 15, 16, 17, 18, 19, 25, 26, 27, 28, 29, 35, 36, 38,
    39, 50, 51, 52, 53, 54, 55, 56, 57, 63, 64, 65, 66, 67, 73, 74
]

tau_stage_0 = np.array([])

tau_stage_1 = np.array([ #2 areas
    24, 62   # PHC
])

tau_stage_2 = np.array([ #14 areas
    2, 40,       # AMYG
    9, 47,       # HC
    30, 68,      # TCC
    31, 69,      # TCI
    32, 70,      # TCPOL
    33, 71,      # TCS
    34, 72       # TCV
])

tau_stage_3 = np.array([ #16 areas
    0, 38,    # A1
    1, 39,    # A2
    11, 49,   # IP
    12, 50,   # M1
    13, 51,   # PCI
    14, 52,   # PCIP
    15, 53,   # PCM
    16, 54,   # PCS
    17, 55,   # PFCCL
    18, 56,   # PFCDL
    19, 57,   # PFCDM
    21, 59,   # PFCORB
    22, 60,   # PFCPOL
    23, 61,   # PFCVL
    25, 63,   # PMCDL
    26, 64,   # PMCM
    27, 65,   # PMCVL
    28, 66,   # S1
    29, 67,   # S2
    35, 73,   # V1
    36, 74    # V2
])

tau_stage_3_b = np.array([
    18, 56,   # PFCDL
    17, 55,   # PFCCL
    19, 57,   # PFCDM
    12, 50,   # M1
    28, 66,   # S1
    29, 67,   # S2
    35, 73,   # V1 
    36, 74    # V2
])

tau_stage_all = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 20, 21, 22, 23, 25, 26, 27, 32, 33, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 51, 52, 53, 54, 58, 59, 60, 61, 63, 64, 65, 70, 71, 75
])

impaired_regions = np.array([21,22,30,31,32,34,59,60,68,69,70,72])

# Stage definitions
tau_stages = np.array([tau_stage_1, tau_stage_2, tau_stage_3], dtype=object)
w_ij_stages = [0.28, 0.78, 0.95]
w_ii_stages = [0.28, 0.78, 0.95]

matrix = get_conn()
tau_matrix_0 = matrix.weights.copy()
tau_matrix_1 = matrix.weights.copy()
tau_matrix_2 = matrix.weights.copy()
tau_matrix_3 = matrix.weights.copy()
tau_matrix_4 = matrix.weights.copy()
tau_matrix_5 = matrix.weights.copy()

deg_perc = [0.00, 0.28, 0.78, 0.95]
w_ii_delta = 0.0127/0.0181
w_ij_delta = 0.0134/0.0192

b_hc = 0.06293568322880852
b_variations = b_hc*(0.3133/0.0593 -1)
stage_0_b = np.ones(76)*b_hc
stage_1_b = np.ones(76)*b_hc
stage_2_b = np.ones(76)*b_hc
stage_3_b = np.ones(76)*b_hc
stage_4_b = np.ones(76)*b_hc
stage_5_b = np.ones(76)*b_hc

a_hc = 0.0728044836955774
a_variations = a_hc*(3.65/3.25 -1)
stage_0_a = np.ones(76)*a_hc
stage_1_a = np.ones(76)*a_hc
stage_2_a = np.ones(76)*a_hc
stage_3_a = np.ones(76)*a_hc
stage_4_a = np.ones(76)*a_hc
stage_5_a = np.ones(76)*a_hc

for i in Abeta_stage_1:
    stage_1_a[i] = a_hc + a_variations*deg_perc[1]
    stage_2_a[i] = a_hc + a_variations*deg_perc[2]
    stage_3_a[i] = a_hc + a_variations*deg_perc[3]
    stage_4_a[i] = a_hc + a_variations*deg_perc[3]
    stage_5_a[i] = a_hc + a_variations*deg_perc[3]
    stage_1_b[i] = b_hc + b_variations*deg_perc[1]
    stage_2_b[i] = b_hc + b_variations*deg_perc[2]
    stage_3_b[i] = b_hc + b_variations*deg_perc[3]
    stage_4_b[i] = b_hc + b_variations*deg_perc[3]
    stage_5_b[i] = b_hc + b_variations*deg_perc[3]

for i in Abeta_stage_2:
    stage_2_a[i] = a_hc + a_variations*deg_perc[1]
    stage_3_a[i] = a_hc + a_variations*deg_perc[2]
    stage_4_a[i] = a_hc + a_variations*deg_perc[3]
    stage_5_a[i] = a_hc + a_variations*deg_perc[3]
    stage_2_b[i] = b_hc + b_variations*deg_perc[1]
    stage_3_b[i] = b_hc + b_variations*deg_perc[2]
    stage_4_b[i] = b_hc + b_variations*deg_perc[3]
    stage_5_b[i] = b_hc + b_variations*deg_perc[3]

for i in Abeta_stage_3:
    stage_3_a[i] = a_hc + a_variations*deg_perc[1]
    stage_4_a[i] = a_hc + a_variations*deg_perc[2]
    stage_5_a[i] = a_hc + a_variations*deg_perc[3]
    stage_3_b[i] = b_hc + b_variations*deg_perc[1]
    stage_4_b[i] = b_hc + b_variations*deg_perc[2]
    stage_5_b[i] = b_hc + b_variations*deg_perc[3]
"""

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

tau_stage_1 = np.array([ #2 areas
    24, 62  # PHC (parahippocampal gyrus)
])

tau_stage_2 = np.array([ #16 areas
    2, 40,   # AMYG (amygdala)
    9, 47,   # HC (hippocampus)
    13, 51,  # PCI (inferor parietal cortex)
    30, 68,  # TCC (central temporal)
    31, 69,  # TCI (inferior temporal)
    32, 70,  # TCPOL (temporal pole)
    33, 71,  # TCS (superior temporal)
    34, 72   # TCV (ventral temporal)
])

tau_stage_3 = np.array([ #34 areas
    0, 38,   # A1 (primary uditivity cortex)
    11, 49,  # IP (intraparietal cortex)
    12, 50,  # M1 (primary motor cortex)
    14, 52,  # PCIP (posterior inferor parietal cortex)
    15, 53,  # PCM (medial parietal cortex)
    16, 54,  # PCS (parietal sulcus)
    17, 55,  # PFCCL (caudolateral prefrontal cortex)
    18, 56,  # PFCDL (dorsolateral prefrontal cortex)
    19, 57,  # PFCDM (dorsomedial prefrontal cortex)
    21, 59,  # PFCORB (orbitofrontal)
    22, 60,  # PFCPOL (prefrontal polar)
    23, 61,  # PFCVL (ventrolateral prefrontal cortex)
    25, 63,  # PMCDL (ventrolateral premotor cortex)
    26, 64,  # PMCM (medial premotor cortex)
    27, 65,  # PMCVL (ventrolaterl premotor cortex)
    28, 66,  # S1 (primary somatosensorial cortex)
    35, 73,  # V1 (primary visul cortex)
])

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
print(tau_matrix_0[24:31, 24:31])
print(tau_matrix_1[24:31, 24:31])
print(tau_matrix_2[24:31, 24:31])
print(tau_matrix_3[24:31, 24:31])
print(tau_matrix_4[24:31, 24:31])
print(tau_matrix_5[24:31, 24:31])

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
np.save(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Degeneration_data\tau_matrices.npy", tau_matrices)
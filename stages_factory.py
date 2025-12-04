#This file only purpose is to one time create a matrix for each parameter (a and b of JR model). Each of this matrix has 4 raws corresponding sto HC, 
# and AD stages 1, 2 and 3 and 76 columns corresponding to the 76 connectivity areas.

import numpy as np
import math

from global_variable_creation import get_conn

Abeta_stage_0 = []

Abeta_stage_1 = [
    22, 60,   # PFCPOL (prefrontal polar)
    21, 59,   # PFCORB (orbitofrontal)
    31, 69,   # TCI (inferior temporal)
    30, 68,   # TCC (central temporal)
    34, 72,   # TCV (ventral temporal)
    32, 70    # TCPOL (temporal pole)
]

Abeta_stage_2 = [
    9, 47,      # Hippocampus (HC)
    24, 62,     # Parahippocampal gyrus (PHC)
    2, 40,      # Amygdala (AMYG)
    10, 48,     # Insula (IA) = anterior + posterior
    20, 58,     # Subgenual cingulum (PFCM)
    5, 43,      # Retrosplenial cingulum (CCR)
    23, 61      # Ventrolateral prefrontal cortex (PFCVL)
]

Abeta_stage_3 = [
    0,  1,  3,  4,  6,  7,  8, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    25, 26, 27, 28, 29, 33, 35, 36, 37,
    38, 39, 41, 42, 44, 45, 46, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    63, 64, 65, 66, 67, 71, 73, 74, 75
]

tau_stage_0 = np.array([])

tau_stage_1 = np.array([
    24, 62   # PHC
])

tau_stage_2 = np.array([
    9, 47,       # Temporal allocortex / hippocampal limbic III -> HC
    31, 69,      # TCI
    30, 68,      # TCC
    34, 72       # TCV
])

tau_stage_3 = np.array([
    18, 56,   # PFCDL
    17, 55,   # PFCCL
    19, 57,   # PFCDM
    12, 50,   # M1
    28, 66,   # S1
    29, 67,   # S2
    35, 73,   # V1 
    36, 74    # V2
])

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

for i in Abeta_stage_2:
    stage_2_a[i] = a_hc + a_variations*deg_perc[1]
    stage_3_a[i] = a_hc + a_variations*deg_perc[2]
    stage_4_a[i] = a_hc + a_variations*deg_perc[3]
    stage_5_a[i] = a_hc + a_variations*deg_perc[3]

for i in Abeta_stage_3:
    stage_3_a[i] = a_hc + a_variations*deg_perc[1]
    stage_4_a[i] = a_hc + a_variations*deg_perc[2]
    stage_5_a[i] = a_hc + a_variations*deg_perc[3]

a_stages = np.array([stage_0_a, stage_1_a, stage_2_a, stage_3_a, stage_4_a, stage_5_a])

for i in tau_stage_1:
    tau_matrix_1[i,:] = tau_matrix_0[i,:] * (1-deg_perc[1]*w_ij_delta)
    tau_matrix_1[:,i] = tau_matrix_0[:,i] * (1-deg_perc[1]*w_ij_delta)
    tau_matrix_1[i,i] = tau_matrix_0[i,i] * (1-deg_perc[1]*w_ii_delta)
    stage_1_b[i] = b_hc + b_variations*deg_perc[1]
    tau_matrix_2[i,:] = tau_matrix_0[i,:] * (1-deg_perc[2]*w_ij_delta)
    tau_matrix_2[:,i] = tau_matrix_0[:,i] * (1-deg_perc[2]*w_ij_delta)
    tau_matrix_2[i,i] = tau_matrix_0[i,i] * (1-deg_perc[2]*w_ii_delta)
    stage_2_b[i] = b_hc + b_variations*deg_perc[2]
    tau_matrix_3[i,:] = tau_matrix_0[i,:] * (1-deg_perc[3]*w_ij_delta)
    tau_matrix_3[:,i] = tau_matrix_0[:,i] * (1-deg_perc[3]*w_ij_delta)
    tau_matrix_3[i,i] = tau_matrix_0[i,i] * (1-deg_perc[3]*w_ii_delta)
    stage_3_b[i] = b_hc + b_variations*deg_perc[3]
    tau_matrix_4[i,:] = tau_matrix_0[i,:] * (1-deg_perc[3]*w_ij_delta)
    tau_matrix_4[:,i] = tau_matrix_0[:,i] * (1-deg_perc[3]*w_ij_delta)
    tau_matrix_4[i,i] = tau_matrix_0[i,i] * (1-deg_perc[3]*w_ii_delta)
    stage_4_b[i] = b_hc + b_variations*deg_perc[3]
    tau_matrix_5[i,:] = tau_matrix_0[i,:] * (1-deg_perc[3]*w_ij_delta)
    tau_matrix_5[:,i] = tau_matrix_0[:,i] * (1-deg_perc[3]*w_ij_delta)
    tau_matrix_5[i,i] = tau_matrix_0[i,i] * (1-deg_perc[3]*w_ii_delta)
    stage_5_b[i] = b_hc + b_variations*deg_perc[3]

for i in tau_stage_2:    
    tau_matrix_2[i,:] = tau_matrix_0[i,:] * (1-deg_perc[1]*w_ij_delta)
    tau_matrix_2[:,i] = tau_matrix_0[:,i] * (1-deg_perc[1]*w_ij_delta)
    tau_matrix_2[i,i] = tau_matrix_0[i,i] * (1-deg_perc[1]*w_ii_delta)
    stage_2_b[i] = b_hc + b_variations*deg_perc[1]
    tau_matrix_3[i,:] = tau_matrix_0[i,:] * (1-deg_perc[2]*w_ij_delta)
    tau_matrix_3[:,i] = tau_matrix_0[:,i] * (1-deg_perc[2]*w_ij_delta)
    tau_matrix_3[i,i] = tau_matrix_0[i,i] * (1-deg_perc[2]*w_ii_delta)
    stage_3_b[i] = b_hc + b_variations*deg_perc[2]
    tau_matrix_4[i,:] = tau_matrix_0[i,:] * (1-deg_perc[3]*w_ij_delta)
    tau_matrix_4[:,i] = tau_matrix_0[:,i] * (1-deg_perc[3]*w_ij_delta)
    tau_matrix_4[i,i] = tau_matrix_0[i,i] * (1-deg_perc[3]*w_ii_delta)
    stage_4_b[i] = b_hc + b_variations*deg_perc[3]
    tau_matrix_5[i,:] = tau_matrix_0[i,:] * (1-deg_perc[3]*w_ij_delta)
    tau_matrix_5[:,i] = tau_matrix_0[:,i] * (1-deg_perc[3]*w_ij_delta)
    tau_matrix_5[i,i] = tau_matrix_0[i,i] * (1-deg_perc[3]*w_ii_delta)
    stage_5_b[i] = b_hc + b_variations*deg_perc[3]

for i in tau_stage_3:
    tau_matrix_3[i,:] = tau_matrix_0[i,:] * (1-deg_perc[1]*w_ij_delta)
    tau_matrix_3[:,i] = tau_matrix_0[:,i] * (1-deg_perc[1]*w_ij_delta)
    tau_matrix_3[i,i] = tau_matrix_0[i,i] * (1-deg_perc[1]*w_ii_delta)
    stage_3_b[i] = b_hc + b_variations*deg_perc[1]
    tau_matrix_4[i,:] = tau_matrix_0[i,:] * (1-deg_perc[2]*w_ij_delta)
    tau_matrix_4[:,i] = tau_matrix_0[:,i] * (1-deg_perc[2]*w_ij_delta)
    tau_matrix_4[i,i] = tau_matrix_0[i,i] * (1-deg_perc[2]*w_ii_delta)
    stage_4_b[i] = b_hc + b_variations*deg_perc[2]
    tau_matrix_5[i,:] = tau_matrix_0[i,:] * (1-deg_perc[3]*w_ij_delta)
    tau_matrix_5[:,i] = tau_matrix_0[:,i] * (1-deg_perc[3]*w_ij_delta)
    tau_matrix_5[i,i] = tau_matrix_0[i,i] * (1-deg_perc[3]*w_ii_delta)
    stage_5_b[i] = b_hc + b_variations*deg_perc[3]

tau_matrices = np.array([tau_matrix_0, tau_matrix_1, tau_matrix_2, tau_matrix_3, tau_matrix_4, tau_matrix_5])
b_stages = np.array([stage_0_b, stage_1_b, stage_2_b, stage_3_b, stage_4_b, stage_5_b])

# Controllo
for i, m in enumerate(tau_matrices, 1):
    print(f"tau_matrix_{i-1}:\n{m[61:70, 61:70]}\n")

for i, m in enumerate(a_stages, 1):
    print(f"a_stage{i-1}:\n{m[61:70]}\n")

for i, m in enumerate(b_stages, 1):
    print(f"b_stage{i-1}:\n{m[61:70]}\n")


#This file only purpose is to one time create a matrix for each parameter (a and b of JR model). Each of this matrix has 4 raws corresponding sto HC, 
# and AD stages 1, 2 and 3 and 76 columns corresponding to the 76 connectivity areas.

import numpy as np
import math

from global_variable_creation import get_conn

tau_stage_0 = np.array([])

tau_stage_1 = [
    24, 62   # PHC
]

tau_stage_2 = [
    9, 47,       # Temporal allocortex / hippocampal limbic III -> HC
    31, 69,      # TCI
    30, 68,      # TCC
    34, 72       # TCV
]

tau_stage_3 = [
    18, 56,   # PFCDL
    17, 55,   # PFCCL
    19, 57,   # PFCDM
    12, 50,   # M1
    28, 66,   # S1
    29, 67,   # S2
    35, 73,   # V1 
    36, 74    # V2
]

def compute_stage_matrix(matrix, stage1=[], stage2=[], stage3=[], deg_perc = [0, 0.05, 0.25, 0.45], w_ij_delta = 1, w_ii_delta = 1, lonely = 1/1.33):
    matrix2 = matrix.copy()
    stages = np.concatenate((stage1, stage2, stage3)).astype(int)
    for i in stages:
        for j in range(matrix.shape[0]):
            if (i in stage1) and (j not in stages):
                matrix[i,j] = matrix2[i,j]*(1-(deg_perc[1] + deg_perc[1]) * w_ii_delta * lonely)
                matrix[j,i] = matrix2[j,i]*(1-(deg_perc[1] + deg_perc[1]) * w_ii_delta * lonely)
            if (i in stage2) and (j not in stages):
                matrix[i,j] = matrix2[i,j]*(1-(deg_perc[2] + deg_perc[2]) * w_ii_delta * lonely)
                matrix[j,i] = matrix2[j,i]*(1-(deg_perc[2] + deg_perc[2]) * w_ii_delta * lonely)
            if (i in stage3) and (j not in stages):
                matrix[i,j] = matrix2[i,j]*(1-(deg_perc[3] + deg_perc[3]) * w_ii_delta * lonely)
                matrix[j,i] = matrix2[j,i]*(1-(deg_perc[3] + deg_perc[3]) * w_ii_delta * lonely)
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

tau_stage_1 = np.array(tau_stage_1, dtype=int)
tau_stage_2 = np.array(tau_stage_2, dtype=int)
tau_stage_3 = np.array(tau_stage_3, dtype=int)

matrix = get_conn()
tau_matrix_0 = matrix.weights.copy()
tau_matrix_1 = matrix.weights.copy()
tau_matrix_2 = matrix.weights.copy()
tau_matrix_3 = matrix.weights.copy()
tau_matrix_4 = matrix.weights.copy()
tau_matrix_5 = matrix.weights.copy()

"""
tau_matrix_5 = compute_stage_matrix(
    tau_matrix_5, 
    stage3 = np.concatenate((tau_stage_1, tau_stage_2, tau_stage_3)).astype(int)
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
# Controllo

for i, m in enumerate(tau_matrices, 1):
    print(f"tau_matrix_{i-1}:\n{m[61:70, 61:70]}\n")
"""

mm = get_conn()
matrix = mm.weights
print(np.array2string(matrix[:10, :10], formatter={'float_kind':lambda x: f"{x:.2f}"}))
matrix = compute_stage_matrix(
    matrix, 
    stage3 = [1,2]
    )

print(np.array2string(matrix[:10, :10], formatter={'float_kind':lambda x: f"{x:.1f}"}))
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
    4, 40, 47, 53
]

tau_stage_2 = [ # 17 regions
    7, 13, 14, 16, 18, 31, 37, 38, 39, 41, 45, 65, 66, 67, 80, 84, 85
]

tau_stage_3 = [ # 38 regions
    0, 2, 3, 6, 9, 11, 15, 17, 19, 21, 22, 23, 24, 25, 27, 29, 33, 34, 44, 46, 49, 51, 52,
    55, 58, 62, 64, 68, 70, 71, 72, 73, 74, 75, 76, 81, 82, 83
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
print(tau_matrix_0[4:8, 4:8])
print(tau_matrix_1[4:8, 4:8])
print(tau_matrix_2[4:8, 4:8])
print(tau_matrix_3[4:8, 4:8])
print(tau_matrix_4[4:8, 4:8])
print(tau_matrix_5[4:8, 4:8])

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
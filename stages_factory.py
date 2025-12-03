#This file only purpose is to one time create a matrix for each parameter (a and b of JR model). Each of this matrix has 4 raws corresponding sto HC, 
# and AD stages 1, 2 and 3 and 76 columns corresponding to the 76 connectivity areas.

import numpy as np

from global_variable_creation import get_conn

a_hc = 0.0728044836955774
a_variations = a_hc*(3.65/3.25 -1)
a_stages = [0.05, 0.28, 0.50, 0.78, 0.95]
a_stages = [0.28, 0.78, 0.95]
for i in range(len(a_stages)):
    a_stages[i] = a_hc + a_stages[i]*a_variations

b_hc = 0.06293568322880852
b_variations = b_hc*(0.3133/0.0593 -1)
b_stages = [0.05, 0.28, 0.50, 0.78, 0.95]
for i in range(len(b_stages)):
    b_stages[i] = b_hc + b_stages[i]*b_variations

#print(a_stages)
#print(b_stages)

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

#tau_stages = np.array([tau_stage_0,tau_stage_1,tau_stage_2,tau_stage_3])

deg_a_stages = []

#for i in range(len(tau_stages)):
#    deg_a_stages[i] = np.concatenate

stage_0_a = np.ones(76)*a_hc
stage_1_a = np.ones(76)*a_hc
stage_2_a = np.ones(76)*a_hc
stage_3_a = np.ones(76)*a_hc
stage_4_a = np.ones(76)*a_hc
stage_5_a = np.ones(76)*a_hc

for i in Abeta_stage_1:
    stage_1_a[i] = a_stages[0]

for i in Abeta_stage_1:
    stage_2_a[i] = a_stages[1]
for i in Abeta_stage_2:
    stage_2_a[i] = a_stages[0]

for i in Abeta_stage_1:
    stage_3_a[i] = a_stages[2]
for i in Abeta_stage_2:
    stage_3_a[i] = a_stages[1]
for i in Abeta_stage_3:
    stage_3_a[i] = a_stages[0]

for i in Abeta_stage_1:
    stage_4_a[i] = a_stages[2]
for i in Abeta_stage_2:
    stage_4_a[i] = a_stages[2]
for i in Abeta_stage_3:
    stage_4_a[i] = a_stages[1]

for i in Abeta_stage_1:
    stage_5_a[i] = a_stages[2]
for i in Abeta_stage_2:
    stage_5_a[i] = a_stages[2]
for i in Abeta_stage_3:
    stage_5_a[i] = a_stages[2]

a_values = [stage_0_a, stage_1_a, stage_2_a, stage_3_a, stage_4_a, stage_5_a]
a_values = np.array(a_values)
print(a_values[:,15:25])

stage_0_b = np.ones(76)*b_hc
stage_1_b = np.ones(76)*b_hc
stage_2_b = np.ones(76)*b_hc
stage_3_b = np.ones(76)*b_hc
stage_4_b = np.ones(76)*b_hc
stage_5_b = np.ones(76)*b_hc

for i in tau_stage_1:
    stage_1_b[i] = b_stages[0]

for i in tau_stage_1:
    stage_2_b[i] = b_stages[1]
for i in tau_stage_2:
    stage_2_b[i] = b_stages[0]

for i in tau_stage_1:
    stage_3_b[i] = b_stages[2]
for i in tau_stage_2:
    stage_3_b[i] = b_stages[1]
for i in tau_stage_3:
    stage_3_b[i] = b_stages[0]

for i in tau_stage_1:
    stage_4_b[i] = b_stages[2]
for i in tau_stage_2:
    stage_4_b[i] = b_stages[2]
for i in tau_stage_3:
    stage_4_b[i] = b_stages[1]

for i in tau_stage_1:
    stage_5_b[i] = b_stages[2]
for i in tau_stage_2:
    stage_5_b[i] = b_stages[2]
for i in tau_stage_3:
    stage_5_b[i] = b_stages[2]

b_values = [stage_0_b, stage_1_b, stage_2_b, stage_3_b, stage_4_b, stage_5_b]
b_values = np.array(b_values)
        

"""
for i in range(len(tau_stages)):
    conn = get_conn()
    rows, cols = np.indices(conn.shape)
    impaired_regions = np.concatenate(tau_stages[:i+1])
    for j in impaired_regions:
        mask = (rows == j) | (cols == j)
        conn[mask] 
    
A = np.ones((3, 3))   # matrice 3x3 di 1

# Creazione maschera: True se riga==2 o colonna==2
rows, cols = np.indices(A.shape)
mask = (rows == 2) | (cols == 2)

# Applica il raddoppio
A[mask] *= 2

print(A)

degeneration_matrices = np.zeros((76,76,36))
hc_mat = get_conn()

for i in range(36):
    """





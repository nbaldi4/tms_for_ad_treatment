import numpy as np
tau_stage = 3
abeta_stage = 3
tau_stage_1 = np.array([
    0, 1
])

tau_stage_2 = np.array([
    2, 3
])

tau_stage_3 = np.array([
    4, 5
])

Abeta_stage_1 = np.array([
    0, 1, 2
])

Abeta_stage_2 = np.array([
    3, 4, 5
])

Abeta_stage_3 = np.array([
    6, 7, 8, 9
])

degeneration = [0, 0, 0, 0.10, 0.50, 1, 1, 1]

a_delta = 1
b_delta = 1

a_array = np.ones((10,1))
b_array = np.ones((10,1))*(1+b_delta*0.01*tau_stage)

for i in tau_stage_1:
    b_array[i] = (1+b_delta*0.5*degeneration[tau_stage + 2])
    
for i in tau_stage_2:
    b_array[i] = (1+b_delta*0.5*degeneration[tau_stage + 1])

for i in tau_stage_3:
    b_array[i] = (1+b_delta*0.5*degeneration[tau_stage])

print('b_array = \n',b_array)

for i in Abeta_stage_1:
    a_array[i] = (1+a_delta*degeneration[abeta_stage + 2])
    b_array[i] = b_array[i] + b_delta*0.5*degeneration[abeta_stage + 2]

for i in Abeta_stage_2:
    a_array[i] = (1+a_delta*degeneration[abeta_stage + 1])
    b_array[i] = b_array[i] + b_delta*0.5*degeneration[abeta_stage + 1]

for i in Abeta_stage_3:
    a_array[i] = (1+a_delta*degeneration[abeta_stage])
    b_array[i] = b_array[i] + b_delta*0.5*degeneration[abeta_stage]

print('a_array = \n',a_array)
print('b_array = \n',b_array)

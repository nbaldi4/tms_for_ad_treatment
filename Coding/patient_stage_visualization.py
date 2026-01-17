import pandas as pd
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import global_variable_creation as gv

from data_analysis_utils import real_vs_stages_PSD, real_vs_stages_FC, to_common_indices, macro_fc, analyze_regular_data, analyze_reduced_data


# Rimuove il limite di larghezza della colonna (None = nessun limite)
pd.set_option('display.max_colwidth', None)

abeta_MCI_patients = np.array([1559.0, np.nan, 338.0, 974.0, np.nan, 384.0, np.nan, 1239.0, 758.0, np.nan, np.nan, 1742.0,
                               np.nan, 527.0, np.nan, 702.0, 1170.0, np.nan, 812.0, 1238.0, np.nan, np.nan, 1202.0, 702.0,
                               np.nan, 692.0, 876.0, 1251.0, 490.0, np.nan, 1320.0, 1962.0, np.nan, 1374.0, np.nan, 1572.0,
                               685.0, 698.0, np.nan, 464.0])
tau_MCI_patients = np.array([40.0, np.nan, 128.2, 20.0, np.nan, 30.6, np.nan, 44.0, 125.0, np.nan, np.nan, 53.0, np.nan, 34.0,
                             np.nan, 81.0, 28.0, np.nan, 56.0, 30.0, np.nan, np.nan, 25.0, 40.0, np.nan, 103.0, 158.0, 36.0,
                             103.0, np.nan, 39.4, 102.6, np.nan, 40.4, np.nan, 42.9, 156.9, 83.0, np.nan, 18.2])
abeta_pos = abeta_MCI_patients < 813
tau_pos = tau_MCI_patients > 52
pos_tau_idxs = np.where(tau_pos)[0]
pos_abeta_idxs = np.where(abeta_pos)[0]
mci = np.where(abeta_pos & tau_pos)[0] 
non_nan_idxs = np.where(~np.isnan(tau_MCI_patients))[0]
abeta_neg = abeta_MCI_patients >= 1000
tau_neg = tau_MCI_patients <= 35
neg_tau_idxs = np.where(tau_neg)[0]
neg_abeta_idxs = np.where(abeta_neg)[0]
healty = np.where(abeta_neg & tau_neg)[0]

fc_patients = np.load(gv.MCI_FC_data)
psd_patients = np.load(gv.MCI_PSD_data)

for i in range(len(psd_patients)):
    print(psd_patients[i].shape)

mean_healty_fc = np.mean(fc_patients[healty], axis=0)
mean_mci_fc = np.mean(fc_patients[mci], axis=0)
mean_healty_psd = np.mean(psd_patients[healty], axis=0)
mean_mci_psd = np.mean(psd_patients[mci], axis=0)

mean_mci_psd = mean_mci_psd[:180].astype(float)
if np.max(mean_mci_psd) == 0:
        mean_mci_psd = mean_mci_psd + 1e-12
mean_mci_psd /= np.max(mean_mci_psd)

mean_healty_psd = mean_healty_psd[:180].astype(float)
if np.max(mean_healty_psd) == 0:
        mean_healty_psd = mean_healty_psd + 1e-12
mean_healty_psd /= np.max(mean_healty_psd)

dt_a = np.zeros(len(non_nan_idxs))
d = np.zeros(len(non_nan_idxs))
d_a = np.zeros(len(non_nan_idxs))
t_a = np.zeros(len(non_nan_idxs))
a = np.zeros(len(non_nan_idxs))
t = np.zeros(len(non_nan_idxs))
b = np.zeros(len(non_nan_idxs))
non_nan_tau = np.zeros((len(non_nan_idxs)))
non_nan_abeta = np.zeros((len(non_nan_idxs)))
normalized_max_sum = np.zeros((len(non_nan_idxs)))
condition = np.zeros((len(non_nan_idxs)))
j=0
for i in non_nan_idxs:
    dt_a[j] = np.sum(psd_patients[i,:31])/np.sum(psd_patients[i,32:62])
    d_a[j] = np.sum(psd_patients[i,:23])/np.sum(psd_patients[i,32:64])
    t_a[j] = np.sum(psd_patients[i,24:31])/np.sum(psd_patients[i,32:64])
    d[j] = np.sum(psd_patients[i,:24])
    a[j] = np.sum(psd_patients[i,32:64])
    t[j] = np.sum(psd_patients[i,24:31])
    b[j] = np.sum(psd_patients[i,65:])
    non_nan_tau[j] = tau_MCI_patients[i]
    non_nan_abeta[j] = abeta_MCI_patients[i]
    normalized_max_sum[j] = (tau_MCI_patients[i] - 52)/np.max(tau_MCI_patients[non_nan_idxs]) - (abeta_MCI_patients[i] - 813)/np.max(abeta_MCI_patients[non_nan_idxs])
    condition[j] = abeta_MCI_patients[i] - (373 + 0.82 * tau_MCI_patients[i])
    j +=1

print('delta+theta over alpha power', np.mean(dt_a), '\pm', np.std(dt_a))
print('delta over alpha power', np.mean(d_a), '\pm', np.std(d_a))
print('theta over alpha power', np.mean(t_a), '\pm', np.std(t_a))
print('delta power', np.mean(d), '\pm', np.std(d))
print('alpha power', np.mean(a), '\pm', np.std(a))
print('theta power', np.mean(t), '\pm', np.std(t))
print('beta-gamma power', np.mean(b), '\pm', np.std(b))

plt.figure(figsize=(10,4))
plt.scatter(non_nan_tau, d, color='r', linewidth=2, )
plt.xlabel("Tau level [mg/L]")
plt.ylabel("delta power")
plt.title(f"Band power over protein level")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10,4))
plt.scatter(non_nan_tau, a, color='r', linewidth=2, )
plt.xlabel("Tau level [mg/L]")
plt.ylabel("alpha power")
plt.title(f"Band power over protein level")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10,4))
plt.scatter(non_nan_tau, t, color='r', linewidth=2, )
plt.xlabel("Tau level [mg/L]")
plt.ylabel("theta power")
plt.title(f"Band power over protein level")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10,4))
plt.scatter(non_nan_abeta, d, color='r', linewidth=2, )
plt.xlabel("ABeta level [mg/L]")
plt.ylabel("delta power")
plt.title(f"Band power over protein level")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10,4))
plt.scatter(non_nan_abeta, a, color='r', linewidth=2, )
plt.xlabel("ABeta level [mg/L]")
plt.ylabel("alpha power")
plt.title(f"Band power over protein level")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10,4))
plt.scatter(non_nan_abeta, t, color='r', linewidth=2, )
plt.xlabel("ABeta level [mg/L]")
plt.ylabel("theta power")
plt.title(f"Band power over protein level")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10,4))
plt.scatter(non_nan_tau, dt_a, color='r', linewidth=2, )
plt.xlabel("Tau level [mg/L]")
plt.ylabel("delta+theta over alpha")
plt.title(f"Relative bands power")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10,4))
plt.scatter(non_nan_abeta, dt_a, color='r', linewidth=2, )
plt.xlabel("ABeta level [mg/L]")
plt.ylabel("delta+theta over alpha")
plt.title(f"Relative bands power")
plt.legend()
plt.grid()
plt.show()
"""
plt.figure(figsize=(10,4))
plt.scatter(condition, dt_a)
plt.grid()
plt.show()
import matplotlib.pyplot as plt
from matplotlib import cm

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 7))

# Usiamo plot_trisurf invece di plot_surface
surf = ax.plot_trisurf(
    non_nan_tau, 
    non_nan_abeta, 
    dt_a, 
    cmap=cm.coolwarm, 
    linewidth=0, 
    antialiased=True
)

ax.set_xlabel('Tau')
ax.set_ylabel('Abeta')
ax.set_zlabel('Ratio DT/A')
ax.set_title('Superficie 3D dei pazienti MCI')

fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()

from scipy.interpolate import griddata

# 1. Definiamo una griglia regolare basata sui minimi e massimi dei dati reali
tau_fine = np.linspace(non_nan_tau.min(), non_nan_tau.max(), 50)
abeta_fine = np.linspace(non_nan_abeta.min(), non_nan_abeta.max(), 50)
tau_grid, abeta_grid = np.meshgrid(tau_fine, abeta_fine)

# 2. Interpoliamo i valori di dt_a sulla griglia
# 'cubic' crea una superficie morbida, 'linear' è più fedele ma spigolosa
dt_a_grid = griddata(
    (non_nan_tau, non_nan_abeta), # Punti originali
    dt_a,                         # Valori originali
    (tau_grid, abeta_grid),       # Nuova griglia
    method='cubic'
)

# 3. Plot della superficie
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 8))
surf = ax.plot_surface(tau_grid, abeta_grid, dt_a_grid, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, alpha=0.8)

# 4. Opzionale: aggiungiamo i punti reali sopra la superficie per vedere dove sono i pazienti
ax.scatter(non_nan_tau, non_nan_abeta, dt_a, color='black', s=20, label='Pazienti reali')

ax.set_xlabel('Tau')
ax.set_ylabel('Abeta')
ax.set_zlabel('dt_a')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
"""
fc_HC_patients = np.load(gv.HC_FC_data)
psd_HC_patients = np.load(gv.HC_PSD_data)
mean_healty_fc = np.mean(fc_HC_patients, axis=0)
mean_healty_psd = np.mean(psd_HC_patients, axis=0)

mean_healty_psd = mean_healty_psd[:180].astype(float)
if np.max(mean_healty_psd) == 0:
        mean_healty_psd = mean_healty_psd + 1e-12
mean_healty_psd /= np.max(mean_healty_psd)

dt_a = np.zeros(len(fc_HC_patients[:,0,0]))
d_a = np.zeros(len(fc_HC_patients[:,0,0]))
t_a = np.zeros(len(fc_HC_patients[:,0,0]))
d = np.zeros(len(fc_HC_patients[:,0,0]))
a = np.zeros(len(fc_HC_patients[:,0,0]))
t = np.zeros(len(fc_HC_patients[:,0,0]))
b = np.zeros(len(fc_HC_patients[:,0,0]))

for i in range(len(fc_HC_patients[:,0,0])):
    dt_a[i] = np.sum(psd_HC_patients[i,:31])/np.sum(psd_HC_patients[i,32:64])
    d_a[i] = np.sum(psd_HC_patients[i,:23])/np.sum(psd_HC_patients[i,32:64])
    t_a[i] = np.sum(psd_HC_patients[i,24:31])/np.sum(psd_HC_patients[i,32:64])
    d[i] = np.sum(psd_HC_patients[i,:23])
    a[i] = np.sum(psd_HC_patients[i,32:64])
    t[i] = np.sum(psd_HC_patients[i,24:31])
    b[i] = np.sum(psd_HC_patients[i,65:])

print('delta+theta over alpha power', np.mean(dt_a), '\pm', np.std(dt_a))
print('delta over alpha power', np.mean(d_a), '\pm', np.std(d_a))
print('theta over alpha power', np.mean(t_a), '\pm', np.std(t_a))
print('delta power', np.mean(d), '\pm', np.std(d))
print('alpha power', np.mean(a), '\pm', np.std(a))
print('theta power', np.mean(t), '\pm', np.std(t))
print('beta-gamma power', np.mean(b), '\pm', np.std(b))

paziente_prova = 22
freqs_uniform = np.linspace(0, 45, 180)
plt.figure(figsize=(10,4))
plt.plot(freqs_uniform, psd_patients[paziente_prova, :180], color='r', linewidth=2, )
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD")
plt.title(f"PSD mci patient {paziente_prova}, delta-theta over alpha = {np.sum(psd_patients[paziente_prova,2:32])/np.sum(psd_patients[paziente_prova,32:])}, tau = {tau_MCI_patients[paziente_prova]}, abeta = {abeta_MCI_patients[paziente_prova]}")
plt.legend()
plt.grid()
plt.show()
"""
plt.figure(figsize=(10,4))
plt.plot(freqs_uniform, mean_mci_psd, color='r', linewidth=2, )
plt.xlabel("Frequency [Hz]")
plt.ylabel("Normalized PSD")
plt.title(f"PSD mci patients")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10,4))
plt.plot(freqs_uniform, mean_healty_psd, color='r', linewidth=2, )
plt.xlabel("Frequency [Hz]")
plt.ylabel("Normalized PSD")
plt.title(f"PSD healty patients")
plt.legend()
plt.grid()
plt.show()

print(mean_healty_psd.shape)

print(pos_tau_idxs)
print(pos_abeta_idxs)
print(mci)
print(healty)
"""
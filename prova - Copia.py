import numpy as np
from scipy.interpolate import interp1d
from fooof import FOOOF
import matplotlib.pyplot as plt
from utils import NSGA_psd, simulate_prova, preprocess, plot_real_vs_simulated_psd_bar, simulate
from data_analysis_utils import NSGA_simulation_vs_real_data_FC
import global_variable_creation as gv

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

def fooof_distance_between_psd(freqs, psd1, psd2):

    # Fit FOOOF models
    fm1 = FOOOF(peak_width_limits=[1, 12])
    fm2 = FOOOF(peak_width_limits=[1, 12])

    fm1.fit(freqs, psd1)
    fm2.fit(freqs, psd2)

    # Aperiodic distance
    ap1 = fm1.aperiodic_params_
    ap2 = fm2.aperiodic_params_
    d_aperiodic = np.linalg.norm(ap1 - ap2)

    # Peak distance
    pk1 = fm1.peak_params_.flatten()
    pk2 = fm2.peak_params_.flatten()
    L = max(len(pk1), len(pk2))
    pk1 = np.pad(pk1, (0, L - len(pk1)))
    pk2 = np.pad(pk2, (0, L - len(pk2)))
    d_peaks = np.linalg.norm(pk1 - pk2)

    return d_aperiodic + d_peaks

ttavg, tavg, teeg, eeg = simulate(g=0.10245373952358368, velocity=41.79385759129675,noise=0.10045650478858126,a=0.09806515133627744,b=0.0680013974341444)
ttavg, tavg = preprocess(ttavg, tavg)
teeg, eeg = preprocess(teeg, eeg)
psd_mae = plot_real_vs_simulated_psd_bar(
        path=r"C:\Users\User\OneDrive - University of Pisa\Desktop\TVB_tutorials\Dati_Healthy\psd_ctr_preview.npy",
        time_series=eeg
    )





simulated_freqs, simulated_psd_mean = NSGA_psd(time_series=eeg)
mask = simulated_freqs <= 45
simulated_freqs = simulated_freqs[mask]
simulated_psd_mean = simulated_psd_mean[mask]
simulated_psd_mean_norm_2 = simulated_psd_mean / np.max(simulated_psd_mean)

path=r"C:\Users\User\OneDrive - University of Pisa\Desktop\TVB_tutorials\Dati_Healthy\psd_ctr_preview.npy"
matrix = np.load(path)
real_psd_mean = np.mean(matrix, axis=0)
real_psd_mean = real_psd_mean[:180]
real_freqs = np.linspace(start=0, stop=0.25*len(real_psd_mean), num=len(real_psd_mean))
simulated_psd_resampled = np.interp(real_freqs, simulated_freqs, simulated_psd_mean_norm_2)
dist = fooof_distance_between_psd(real_freqs, real_psd_mean, simulated_psd_resampled)

"""
fmreal.print_results()
freq_used_real = fmreal.freqs
aperiodic_fit_real = fmreal._ap_fit
psd_used_real =fmreal.power_spectrum
# PSD meno il background aperiodico
psd_aperiodic_removed_real = psd_used_real - aperiodic_fit_real
"""
real_psd_mean_norm = real_psd_mean / np.sum(real_psd_mean)
simulated_psd_mean_norm = simulated_psd_mean / np.sum(simulated_psd_mean)
emd_value = wasserstein_distance(real_freqs, simulated_freqs, real_psd_mean_norm, simulated_psd_mean_norm)

# ----------------------------------------
# 6) Plot confronto PSD
# ----------------------------------------
real_psd_mean_norm_2 = real_psd_mean / np.max(real_psd_mean) 
plt.figure(figsize=(10,5))
plt.plot(real_freqs, real_psd_mean_norm_2, label="real PSD (normalizzata)")
plt.plot(simulated_freqs, simulated_psd_mean_norm_2, label="simulated PSD (normalizzata)")
plt.xlabel("Frequenza (Hz)")
plt.ylabel("Densità normalizzata")
plt.title("Confronto PSD per EMD")
plt.legend()
plt.show()

print("PSD MAE: ", psd_mae)
print(f"PSD \nEarth Mover's Distance (Wasserstein): {emd_value:.4f}")
print("Distanza FOOOF =", dist)

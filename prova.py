import numpy as np
from data_analysis_utils import analyze_reduced_data, analyze_regular_data, real_vs_simulated_PSD, real_vs_simulated_FC
from utils import simulate
import matplotlib.pyplot as plt
import global_variable_creation as gv

MCI_FC_data = np.load(r'C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Dati_MCI\fc_mci_preview.npy')
MCI_PSD_data = np.load(r'C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Dati_MCI\psd_mci_preview.npy')
gv.show_plots = True

ttavg, tavg, teeg, eeg = simulate(g=gv.coup, velocity=gv.vel, noise=gv.noi, a=gv.aa, b=gv.bb, sim_time=3000)

real_vs_simulated_PSD(MCI_PSD_data, eeg)
real_vs_simulated_FC(MCI_FC_data, eeg)
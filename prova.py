import numpy as np
import matplotlib.pyplot as plt
import global_variable_creation as gv

MCI_FC_data = np.load(r'C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Dati_MCI\fc_mci_preview.npy')
MCI_PSD_data = np.load(r'C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Dati_MCI\psd_mci_preview.npy')

matrix = MCI_FC_data[0,:,:]
matrix = np.clip(matrix, -0.999999, 0.999999)
matrix = 2 * (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix)) - 1

plt.title(f"Regular size real FC")
plt.imshow(matrix, cmap='bwr', vmin=-1, vmax=1)
plt.colorbar()   # <-- ora funziona!
plt.show()
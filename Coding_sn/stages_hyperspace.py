import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import colors

no_stim_box = pd.read_pickle(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\No_stim_box_plots\no_stim_box.pkl")
metrics = ['Pdobr', 'Ptobr', 'Paobr', 'Pbobr']#['Fa', 'Fbe', 'Pdobr', 'Ptobr', 'Paobr', 'Pbobr']#, 'Fd', 'Ft', 'Fa', 'Fbe', 'Pdobr', 'Ptobr', 'Paobr', 'Pbobr']
suffix = '_med'

psd_delta = no_stim_box["Pdobr_med"]
psd_theta = no_stim_box["Ptobr_med"]
psd_alpha = no_stim_box["Paobr_med"]
psd_beta  = no_stim_box["Pbobr_med"]
abe_value = no_stim_box["b"]
tau_value = no_stim_box["t"]

points = []
for i in range(len(psd_delta)):
    points.append(np.array([psd_theta[i], psd_alpha[i], psd_beta[i], 2*abe_value[i] + tau_value[i]]))

arr = np.array(points)

a = arr[:, 0] # PSD theta
b = arr[:, 1] # PSD alpha
c = arr[:, 2] # PSD beta 
d = arr[:, 3] # disease value

bounds = [0, 4, 7, 10, 13, 15]

# 3. Creiamo una colormap discreta con 5 livelli di grigio
# 'gray' crea una scala dal nero al bianco; 5 indica il numero di intervalli
cmap = plt.get_cmap('gray', 5)

# 4. Creiamo il "norm" che associa i valori ai bin
norm = colors.BoundaryNorm(bounds, cmap.N)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scat = ax.scatter(a, b, c, c=d, cmap=cmap, norm=norm, s=60, edgecolors='k', alpha=0.8)

# 6. Colorbar personalizzata con i nomi degli intervalli
cbar = fig.colorbar(scat, ax=ax, ticks=[3, 6, 9, 12, 15], label='Disease value')
cbar.ax.set_yticklabels(['0-3', '4-6', '7-9', '10-12', '13-15'])
ax.set_xlabel('PSD theta')
ax.set_ylabel('PSD alpha')
ax.set_zlabel('PSD beta')
ax.set_title('Disease value')

plt.show()
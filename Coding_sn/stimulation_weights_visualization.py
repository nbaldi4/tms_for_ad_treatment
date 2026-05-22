import os

import numpy as np
from tvb.simulator.lab import *
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import stats
import matplotlib.patches as mpatches

def main():

    import pandas as pd
    file_name = f"super_th_reg_a{3}_t{3}_f{10}_s{2}"
    base_dir = os.path.dirname(os.path.abspath(__file__))  # cartella di utils.py
    path = os.path.join(base_dir, '..', 'Tms_results_end', file_name)
    df = pd.read_pickle(path)

    intensity = 10

    intensityplus = intensity + 0.01
    intensityminus = intensity - 0.01
    df=df.query("sf <= @intensityplus and sf >= @intensityminus")
        
    pre_stim_amp = df["mean_amp_pre"].reset_index(drop=True)
    post_stim_amp = df["mean_amp_post"].reset_index(drop=True)
    baseline = df["mean_baseline"].reset_index(drop=True)
    threshold = df["is_greater"].reset_index(drop=True)
    
    norm = 0.2
    second_order_weights = (np.abs(post_stim_amp-pre_stim_amp))/norm

    def create_connectivity86():

        """Factory che gestisce un oggetto Connectivity con configurazione persistente."""
        filename = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\HC_data\real_patient_connectome.zip"
        config = {
            'filename': filename
        }

        # Altrimenti crea una nuova Connectivity
        conn = connectivity.Connectivity.from_file(config['filename'])
        conn.configure()  # opzionale, se serve come per EEG

        return conn

    conn_86 = create_connectivity86()

    def plot_regions(centres, region_values, title, cortex, cortex_simnibs):
        region_values = np.array(region_values, dtype=float).flatten()  # ← dtype=float esplicito
        centres = np.array(centres)  # ← anche centres per sicurezza

        # Indici delle regioni subcorticali (i tuoi zeri: 34-48 + altri)
        zero_indices = np.where(region_values == 0)[0]
        cortical_mask = np.ones(len(centres), dtype=bool)
        cortical_mask[zero_indices] = False  # escludi subcorticali

        # KDTree solo sui centri corticali
        cortical_centres = centres[cortical_mask]
        cortical_indices = np.where(cortical_mask)[0]  # indici originali

        tree = KDTree(cortical_centres)
        _, nearest_local = tree.query(cortex.vertices)
        nearest_region = cortical_indices[nearest_local]  # rimappa agli indici originali
        
        #tree = KDTree(centres)
        cortex.configure()
        x_cortex, y_cortex, z_cortex = cortex.vertices.T

        vertex_values = region_values[nearest_region]
        tri_values = vertex_values[cortex.triangles].mean(axis=1)

        # Colormap continua: giallo->arancione->rosso, grigio per 0
        # Scala lineare, ma colormap che parte da un colore chiaro
        # 'plasma' va da viola scuro a giallo brillante — niente nero, niente bianco
        cmap = plt.cm.plasma  

        norm = plt.Normalize(vmin=0, vmax=region_values.max())

        # Grigio solo per gli zeri esatti
        tri_vertex_values = region_values[nearest_region][cortex.triangles]
        tri_all_zero = (tri_vertex_values == 0).all(axis=1)

        face_colors = cmap(norm(tri_values))
        face_colors[tri_all_zero] = [0.4, 0.4, 0.4, 1.0]

        verts = cortex.vertices[cortex.triangles]

        if cortex_simnibs:
            views = [
                ('Top',        90,  -90),
                ('Bottom',    -90,   90),
                ('Frontal',     0,   90),
                ('Back',        0,  -90),
                ('Lateral sx',  0,    0),
                ('Lateral dx',  0,  180),
            ]
        else:
            views = [
                ('Top',        90, -180),
                ('Bottom',    -90,    0),
                ('Frontal',     0,    0),
                ('Back',        0,  180),
                ('Lateral sx',  0,  -90),
                ('Lateral dx',  0,   90),
            ]

        fig = plt.figure(figsize=(28, 20))

        for i, (view_name, elev, azim) in enumerate(views):
            ax = fig.add_subplot(2, 3, i + 1, projection='3d')

            poly = Poly3DCollection(verts, alpha=0.8, edgecolor='none')
            poly.set_facecolor(face_colors)
            ax.add_collection3d(poly)

            ax.set_xlim(x_cortex.min(), x_cortex.max())  # ← usa x_cortex, non cortex
            ax.set_ylim(y_cortex.min(), y_cortex.max())
            ax.set_zlim(z_cortex.min(), z_cortex.max())

            ax.set_title(view_name, fontsize=100, pad=0)

            ax.tick_params(axis='both', labelsize=70, pad=20)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.zaxis.set_major_locator(plt.MaxNLocator(4))

            ax.view_init(elev=elev, azim=azim)


        plt.tight_layout(rect=[0, 0, 0.85, 1])

        cbar_ax = fig.add_axes([0.87, 0.2, 0.025, 0.6])  # [left, bottom, width, height]
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=40)
        cbar.set_label('stimulus weight', fontsize=50, labelpad=20)

        # Rimuove l'asse spurio 0-1 che appare sotto
        cbar.ax.set_aspect('auto')
        plt.savefig(os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop", title),
                    bbox_inches='tight', dpi=200)
        plt.close()

    cortex_simnibs = r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\sub-CLB00044_Cortex.zip"

    sn_weights = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\Pesi_protocolli_TMS\Fp1_1_stimulus_vector_TVB.npy")
    plot_regions(centres=conn_86.centres, region_values=second_order_weights, cortex_simnibs=True, title='fp1_1mm_indirect_stim', cortex=surfaces.CorticalSurface.from_file(cortex_simnibs))
    
if __name__ == "__main__":
    main()
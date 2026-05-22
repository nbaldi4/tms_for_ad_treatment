import os

import numpy as np
from tvb.simulator.lab import *
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import stats
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colorbar as mcolorbar
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def main():
    tau_stage_1 = [ #2 areas
        24, 62  # PHC (parahippocampal gyrus)
    ]

    tau_stage_2 = [ #16 areas
        2, 40,   # AMYG (amygdala)
        9, 47,   # HC (hippocampus)
        13, 51,  # PCI (inferor parietal cortex)
        30, 68,  # TCC (central temporal)
        31, 69,  # TCI (inferior temporal)
        32, 70,  # TCPOL (temporal pole)
        33, 71,  # TCS (superior temporal)
        34, 72   # TCV (ventral temporal)
    ]

    tau_stage_3 = [ #34 areas
        0, 38,   # A1 (primary uditivity cortex)
        11, 49,  # IP (intraparietal cortex)
        12, 50,  # M1 (primary motor cortex)
        14, 52,  # PCIP (posterior inferor parietal cortex)
        15, 53,  # PCM (medial parietal cortex)
        16, 54,  # PCS (parietal sulcus)
        17, 55,  # PFCCL (caudolateral prefrontal cortex)
        18, 56,  # PFCDL (dorsolateral prefrontal cortex)
        19, 57,  # PFCDM (dorsomedial prefrontal cortex)
        21, 59,  # PFCORB (orbitofrontal)
        22, 60,  # PFCPOL (prefrontal polar)
        23, 61,  # PFCVL (ventrolateral prefrontal cortex)
        25, 63,  # PMCDL (ventrolateral premotor cortex)
        26, 64,  # PMCM (medial premotor cortex)
        27, 65,  # PMCVL (ventrolaterl premotor cortex)
        28, 66,  # S1 (primary somatosensorial cortex)
        35, 73,  # V1 (primary visul cortex)
    ]

    Abeta_stage_1 = [ #12 areas
        21, 59,   # PFCORB (orbitofrontal)
        22, 60,   # PFCPOL (prefrontal polar)
        30, 68,   # TCC (central temporal)
        31, 69,   # TCI (inferior temporal)
        32, 70,   # TCPOL (temporal pole)
        34, 72    # TCV (ventral temporal)
    ]

    Abeta_stage_2 = [ #22 areas
        2, 40,   # AMYG (amygdala)
        3, 41,   # CCA (anterior cingulate cortex)
        4, 42,   # CCP (posterior cingulate cortex)
        5, 43,   # CCR (rostral cingulate cortex)
        6, 44,   # CCS (subgenual cingulate cortex)
        9, 47,   # HC (hippocampus)
        10, 48,  # IA (insula = anterior + posterior)
        23, 61,  # PFCVL (ventrolateral prefrontal cortex)
        24, 62,  # PHC (parahippocampal gyrus)
        33, 71,  # TCS (superior temporal)
        37, 75   # CC (cingulate cortex)
    ]

    Abeta_stage_3 = [ #42 areas
        0, 38,   # A1 (primary uditivity cortex)
        1, 39,   # A2 (secondary uditivity cortex)
        7, 45,   # FEF (frontal eyes field)
        8, 46,   # G (gyrus)
        11, 49,  # IP (intraparietal cortex)
        12, 50,  # M1 (primary motor cortex)
        13, 51,  # PCI (inferor parietal cortex)
        14, 52,  # PCIP (posterior inferor parietal cortex)
        15, 53,  # PCM (medial parietal cortex)
        16, 54,  # PCS (parietal sulcus)
        17, 55,  # PFCCL (caudolateral prefrontal cortex)
        18, 56,  # PFCDL (dorsolateral prefrontal cortex)
        19, 57,  # PFCDM (dorsomedial prefrontal cortex)
        20, 58,  # PFCM (medial prefrontal cortex)
        25, 63,  # PMCDL (ventrolateral premotor cortex)
        26, 64,  # PMCM (medial premotor cortex)
        27, 65,  # PMCVL (ventrolaterl premotor cortex)
        28, 66,  # S1 (primary somatosensorial cortex)
        29, 67,  # S2 (secondary somatosensorial cortex)
        35, 73,  # V1 (primary visul cortex)
        36, 74,  # V2 (secondary visual cortex)
    ]

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
    second_oreder_weights = (np.abs(post_stim_amp-pre_stim_amp))/norm

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

    def create_connectivity76():

        """Factory che gestisce un oggetto Connectivity con configurazione persistente."""
        config = {
            'filename': 'connectivity_76.zip'
        }
        
        # Altrimenti crea una nuova Connectivity
        conn = connectivity.Connectivity.from_file(config['filename'])
        conn.configure()  # opzionale, se serve come per EEG

        return conn

    conn_76 = create_connectivity76()

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

    def plot_dual_protein_sagittal(centres, abeta_values, tau_values, title, cortex, side='left'):
        # Assicuriamoci che siano array
        abeta_values = np.array(abeta_values, dtype=float).flatten()
        tau_values = np.array(tau_values, dtype=float).flatten()
        
        # Normalizzazione (0 to 1)
        # Se il valore massimo è 0 (nessun accumulo), evitiamo la divisione per zero
        norm_abeta = abeta_values / (abeta_values.max() if abeta_values.max() > 0 else 1)
        norm_tau = tau_values / (tau_values.max() if tau_values.max() > 0 else 1)

        # Creazione della matrice RGBA per ogni regione (86, 4)
        region_colors = np.zeros((len(abeta_values), 4))

        
        for i in range(len(abeta_values)):
            if abeta_values[i] == 0 and tau_values[i] == 0:
                # Grigio per regioni sane (bassa saturazione/alpha)
                region_colors[i] = [0.5, 0.5, 0.5, 0.1] 
            else:
                # Canale Rosso = Amiloide, Canale Blu = Tau
                # Usiamo l'alpha (trasparenza) massimo a 0.5 come richiesto
                r = norm_abeta[i]
                g = 0
                b = norm_tau[i]
                # L'alpha scala con l'accumulo totale, ma cappato a 0.5
                a = max(norm_abeta[i], norm_tau[i]) * 0.5
                if a < 0.1: a = 0.1 # Minimo di visibilità
                
                region_colors[i] = [r, g, b, a]

        # --- LOGICA DI FILTRAGGIO EMISFERO (come prima) ---
        x_coords = cortex.vertices[:, 1]
        vertex_mask = x_coords <= 0 if side == 'left' else x_coords >= 0
        faces_mask = np.all(vertex_mask[cortex.triangles], axis=1)
        filtered_triangles = cortex.triangles[faces_mask]
        #filtered_triangles = cortex.triangles

        # Mappatura dai centri ai vertici
        tree = KDTree(centres)
        _, nearest_region = tree.query(cortex.vertices)
        
        # Colore dei vertici basato sulla miscela RGBA
        vertex_colors = region_colors[nearest_region]
        # Colore delle facce (media dei colori dei vertici)
        face_colors = vertex_colors[filtered_triangles].mean(axis=1)

        # --- PLOT ---
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        fig.subplots_adjust(left=-0.1, right=1.1, bottom=-0.1, top=1.1)
        
        verts = cortex.vertices[filtered_triangles]
        poly = Poly3DCollection(verts, edgecolor='none')
        poly.set_facecolor(face_colors) # Accetta direttamente l'array RGBA
        ax.add_collection3d(poly)

        # Colorbar Amiloide (rosso)

        #norm_abeta_scale = mcolors.Normalize(vmin=0, vmax=abeta_values.max() if abeta_values.max() > 0 else 1)
        #sm_abeta = cm.ScalarMappable(cmap=cm.Reds, norm=norm_abeta_scale)
        #sm_abeta.set_array([])
        #cbar_abeta = fig.colorbar(sm_abeta, ax=ax, shrink=0.4, pad=0.02, location='left')
        #cbar_abeta.set_label('abeta', fontsize=11)
        # Colorbar Tau (blu)
        #norm_tau_scale = mcolors.Normalize(vmin=0, vmax=tau_values.max() if tau_values.max() > 0 else 1)
        #sm_tau = cm.ScalarMappable(cmap=cm.Blues, norm=norm_tau_scale)
        #sm_tau.set_array([])
        #cbar_tau = fig.colorbar(sm_tau, ax=ax, shrink=0.4, pad=0.02, location='right')
        #cbar_tau.set_label('tau', fontsize=11)

        # Limiti e vista
        #filtered_verts = cortex.vertices[vertex_mask]
        ax.set_xlim(cortex.vertices[:,0].min(), cortex.vertices[:,0].max())
        ax.set_ylim(cortex.vertices[:,1].min(), cortex.vertices[:,1].max())
        ax.set_zlim(cortex.vertices[:,2].min(), cortex.vertices[:,2].max())
        ax.set_axis_off()
        #ax.view_init(elev=90, azim=-90)
        ax.view_init(elev=0, azim=90 if side=='left' else -90)

        # Forza il rapporto d'aspetto basato sui dati reali
        x_range = cortex.vertices[:,0].max() - cortex.vertices[:,0].min()
        y_range = cortex.vertices[:,1].max() - cortex.vertices[:,1].min()
        z_range = cortex.vertices[:,2].max() - cortex.vertices[:,2].min()
        ax.set_box_aspect([x_range, y_range, z_range])

        plt.tight_layout()
       
        from PIL import Image
        import io

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=450, bbox_inches='tight', pad_inches=0, transparent=True)
        buf.seek(0)

        img = Image.open(buf)
        # Converte in RGBA e ritaglia i pixel non-bianchi
        img_array = np.array(img)
        non_white = np.where(img_array[:,:,:3] < 255)  # pixel non bianchi
        if len(non_white[0]) > 0:
            y_min, y_max = non_white[0].min(), non_white[0].max()
            x_min, x_max = non_white[1].min(), non_white[1].max()
            margin = 10  # pixel di margine
            cropped = img.crop((
                max(0, x_min - margin),
                max(0, y_min - margin),
                min(img.width, x_max + margin),
                min(img.height, y_max + margin)
            ))
            cropped.save(os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop\Thesis\Immagini_tesi\Results\Stages_grid_half_brain", title))
            plt.close()


        #plt.savefig(os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop", title), bbox_inches='tight', dpi=300, pad_inches=0)
        #plt.show()

    
    cortex_simnibs = r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\sub-CLB00044_Cortex.zip"

    sn_weights = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\Pesi_protocolli_TMS\Fp1_1_stimulus_vector_TVB.npy")
    #plot_regions(centres=conn_86.centres, region_values=second_oreder_weights, cortex_simnibs=True, title='fp1_1mm_indirect_stim', cortex=surfaces.CorticalSurface.from_file(cortex_simnibs))
    abeta_0 = np.zeros(76)
    tau_0 = np.zeros(76)
    abeta_1 = np.zeros(76)
    tau_1 = np.zeros(76)
    abeta_2 = np.zeros(76)
    tau_2 = np.zeros(76)
    abeta_3 = np.zeros(76)
    tau_3 = np.zeros(76)
    abeta_4 = np.zeros(76)
    tau_4 = np.zeros(76)
    abeta_5 = np.zeros(76)
    tau_5 = np.zeros(76)
    for i in Abeta_stage_1:
        abeta_1[i]=1/3
        abeta_2[i]=2/3
        abeta_3[i]=1
        abeta_4[i]=1
        abeta_5[i]=1
    for i in Abeta_stage_2:
        abeta_1[i]=0
        abeta_2[i]=1/3
        abeta_3[i]=2/3
        abeta_4[i]=1
        abeta_5[i]=1
    for i in Abeta_stage_3:
        abeta_1[i]=0
        abeta_2[i]=0
        abeta_3[i]=1/3
        abeta_4[i]=2/3
        abeta_5[i]=1
        
    for i in tau_stage_1:
        tau_1[i]=1/3
        tau_2[i]=2/3
        tau_3[i]=1
        tau_4[i]=1
        tau_5[i]=1
    for i in tau_stage_2:
        tau_1[i]=0
        tau_2[i]=1/3
        tau_3[i]=2/3
        tau_4[i]=1
        tau_5[i]=1
    for i in tau_stage_3:
        tau_1[i]=0
        tau_2[i]=0
        tau_3[i]=1/3
        tau_4[i]=2/3
        tau_5[i]=1
    
    abetas = np.array((abeta_0, abeta_1, abeta_2, abeta_3, abeta_4, abeta_5))
    taus = np.array((tau_0, tau_1, tau_2, tau_3, tau_4, tau_5))

    for i in range(len(abetas)):
        for j in range(len(taus)):
            title = f'abeta_{i}_tau_{j}_no_colorbar.png'
            #print('abeta stage: ', i)
            #print('tau stage: ', j)
            #abb = abetas[i]
            #tta = taus[j]
            #print('abeta values: ',abb[21:30])
            #print('tau values: ',tta[21:30])
            plot_dual_protein_sagittal(centres=conn_76.centres, abeta_values=abetas[i], tau_values=taus[j], cortex=surfaces.CorticalSurface.from_file(), title=title)
    #plot_dual_protein_sagittal(centres=conn_76.centres, abeta_values=abeta_5, tau_values=tau_5, cortex=surfaces.CorticalSurface.from_file(), title='abeta_5_tau_5.png')

if __name__ == "__main__":
    main()
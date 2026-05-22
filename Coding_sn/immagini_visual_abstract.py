import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tvb.simulator.lab import *
from scipy.spatial import KDTree
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd


file_name = f"super_th_reg_a{3}_t{3}_f{10}_s{6}"
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

def fc_alpha():
    # --- 1. Caricamento dati ---
    fc_alpha_tavg_00 = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Stages_simnibs_2\Stage_0_0\tau0_abeta0_a0.0712_b0.0547_g5.2173_v64.22_n0.0407_simtime20000_rs0\FC_TAVG alpha_tau_0_abeta_0_band_8-13_Hz.npz")
    fc_reg_00 = fc_alpha_tavg_00['normal_size']

    fc_alpha_tavg_44 = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Stages_simnibs_2\Stage_4_4\tau4_abeta4_a0.0712_b0.0547_g5.2173_v64.22_n0.0407_simtime20000_rs0\FC_TAVG alpha_tau_4_abeta_4_band_8-13_Hz.npz")
    fc_reg_44 = fc_alpha_tavg_44['normal_size']

    # --- 2. Configurazione della Soglia Sicura e dei Colori ---
    soglia = 0.148

    min_assoluto = min(fc_reg_00.min(), fc_reg_44.min())
    max_assoluto = max(fc_reg_00.max(), fc_reg_44.max())
    soglia_sicura = max(min_assoluto + 1e-5, min(max_assoluto - 1e-5, soglia))

    cmap_bicolore = mcolors.ListedColormap(['blue', 'red'])
    limiti = [min_assoluto, soglia_sicura, max_assoluto]
    norm_soglia = mcolors.BoundaryNorm(limiti, cmap_bicolore.N)


    # --- 3. GRAFICO 1: Stage 0_0 (Finestra Indipendente) ---
    fig1, ax1 = plt.subplots(figsize=(7, 7)) # Crea la prima finestra quadrata
    im1 = ax1.imshow(fc_reg_00, cmap=cmap_bicolore, norm=norm_soglia, interpolation='nearest')
    #ax1.set_title("Stage 0_0", fontsize=14, pad=15)
    ax1.axis('off') # Nessun tick, nessuna etichetta


    # --- 4. GRAFICO 2: Stage 4_4 (Finestra Indipendente) ---
    fig2, ax2 = plt.subplots(figsize=(7, 7)) # Crea la seconda finestra quadrata
    im2 = ax2.imshow(fc_reg_44, cmap=cmap_bicolore, norm=norm_soglia, interpolation='nearest')
    #ax2.set_title("Stage 4_4", fontsize=14, pad=15)
    ax2.axis('off') # Nessun tick, nessuna etichetta

    cartella_output = r"C:\Users\User\OneDrive - University of Pisa\Desktop\Thesis\Visual_abstract\Immagini_che_ha_chiesto_lorenzo_il_16_05\\"

    # --- 6. Salvataggio delle figure ---
    # Salviamo la prima figura (Stage 0_0)
    fig1.savefig(cartella_output + "fc_alpha_stage_0_0_th_0148.png", dpi=450, bbox_inches='tight', pad_inches=0, transparent=True)

    # Salviamo la seconda figura (Stage 4_4)
    fig2.savefig(cartella_output + "fc_alpha_stage_4_4_th_0148.png", dpi=450, bbox_inches='tight', pad_inches=0, transparent=True)
    # --- 5. Mostra entrambe le finestre ---
    plt.show()


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

def plot_dual_protein_sagittal(centres, weights, cortex, title, side='left'):
    combined = np.array(weights, dtype=float).flatten()

    # Normalizzazione 0→1
    combined_max = combined.max() if combined.max() > 0 else 1
    norm_combined = combined / combined_max

    # Mappatura colore con plasma
    cmap = plt.cm.plasma
    region_colors = cmap(norm_combined)  # shape (N, 4) RGBA

    # Regioni sane → grigio trasparente
    for i in range(len(combined)):
        if combined[i] == 0:
            region_colors[i] = [0.5, 0.5, 0.5, 0.15]

    filtered_triangles = cortex.triangles

    # Mappatura centri → vertici con KDTree
    tree = KDTree(centres)
    _, nearest_region = tree.query(cortex.vertices)

    vertex_colors = region_colors[nearest_region]
    face_colors = vertex_colors[filtered_triangles].mean(axis=1)

    # --- PLOT ---
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=-0.1, right=1.1, bottom=-0.1, top=1.1)

    verts = cortex.vertices[filtered_triangles]
    poly = Poly3DCollection(verts, edgecolor='none')
    poly.set_facecolor(face_colors)
    ax.add_collection3d(poly)

    # Limiti e vista
    ax.set_xlim(cortex.vertices[:,0].min(), cortex.vertices[:,0].max())
    ax.set_ylim(cortex.vertices[:,1].min(), cortex.vertices[:,1].max())
    ax.set_zlim(cortex.vertices[:,2].min(), cortex.vertices[:,2].max())
    ax.set_axis_off()
    ax.view_init(elev=0, azim=180 if side=='left' else 0)


    x_range = cortex.vertices[:,0].max() - cortex.vertices[:,0].min()
    y_range = cortex.vertices[:,1].max() - cortex.vertices[:,1].min()
    z_range = cortex.vertices[:,2].max() - cortex.vertices[:,2].min()
    ax.set_box_aspect([x_range, y_range, z_range])

    plt.tight_layout()

    from PIL import Image
    import io

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=450, bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    img = Image.open(buf)
    img_array = np.array(img)
    non_white = np.where(img_array[:,:,:3] < 255)
    if len(non_white[0]) > 0:
        y_min, y_max = non_white[0].min(), non_white[0].max()
        x_min, x_max = non_white[1].min(), non_white[1].max()
        margin = 10
        cropped = img.crop((
            max(0, x_min - margin),
            max(0, y_min - margin),
            min(img.width, x_max + margin),
            min(img.height, y_max + margin)
        ))
        cropped.save(os.path.join(
            r"C:\Users\User\OneDrive - University of Pisa\Desktop\Thesis\Visual_abstract\Immagini_che_ha_chiesto_lorenzo_il_16_05",
            title+'.png'
        ))
    plt.close()

cortex_simnibs = r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\sub-CLB00044_Cortex.zip"
second_oreder_weights = second_oreder_weights/np.max(second_oreder_weights)
plot_dual_protein_sagittal(
     centres=conn_86.centres,
     weights = second_oreder_weights, #np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\Pesi_protocolli_TMS\T7T3_1_stimulus_vector_TVB.npy"),
     cortex= surfaces.CorticalSurface.from_file(cortex_simnibs),
     title='perceived_stimulation_T7_1_mm_right',
     side='right'
     )
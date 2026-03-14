import os

import numpy as np
import argparse
from tvb.simulator.lab import *
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils import skin_and_sensors, brain_activity, simulate, preprocess, eeg_plot, simulate_different_combination_parameters, simulate_parametri_lorenzo
from utils import functional_connectivity, psd, functional_connectivity_band, functional_connectivity_img_coh, psd_simil_lorenzo, Simulate2, functional_connectivity_lorenzo
from utils import make_train, simulate_with_stimulus, stimulus_visualization, new_parcellation_simulate
from classes import MultiStimuliRegion
from data_analysis_utils import real_vs_simulated_FC, real_vs_simulated_PSD, NSGAII, analyze_regular_data, analyze_reduced_data, load_simulation_results
import global_variable_creation as gv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_plots", action="store_true", help="Show plots on screen")
    parser.add_argument("--conn_sim", action="store_true", help="Use 86 regions connectivity")
    parser.add_argument("--no_sim", action="store_true", help="Search for simulated data")
    parser.add_argument("--save_plots", action="store_true", help="Save plots to disk")
    parser.add_argument("--save_folder", type=str, help="Folder to save figures and data")
    parser.add_argument("--save_raw_data", action="store_true", help="Saving raw eeg and tavg data")
    parser.add_argument("--struct_conn", type=str, help="Path to structural connectivity file")
    parser.add_argument("--EEG", action="store_true", help="Perform eeg")
    parser.add_argument("--PSD", action="store_true", help="Perform psd")
    parser.add_argument("--FC", action="store_true", help="Perform fc")
    parser.add_argument("--brain_act", action="store_true", help="Compute brain activity")
    parser.add_argument("--skin_and_sens", action="store_true", help="Visualize skin and sensors positions")
    parser.add_argument("--th", type=float, help="Threshold for FC visualization")
    parser.add_argument("--temp_avg_period", type=float, help="Average integration step")
    parser.add_argument("--sim_vs_real", action="store_true", help="Perform comparison between simulation and real data")
    parser.add_argument("--g", type=float, help="Coupling")
    parser.add_argument("--vel", type=float, help="Conduction velocity")
    parser.add_argument("--noise", type=float, help="Noise")
    parser.add_argument("--a", type=float, help="JR a parameter")
    parser.add_argument("--b", type=float, help="JR b parameter")
    parser.add_argument("--tau", type=int, help="Tau stage (from 0 to 5)")
    parser.add_argument("--abeta", type=int, help="Beta Amyloid stage (from 0 to 5)")
    parser.add_argument("--rnd", type=int, help="Simulation random seed")
    parser.add_argument(
    "--tau_array", 
    type=int, 
    nargs=3, 
    help="Lista degli stadi Tau (es: --tau_array 0 0 0)"
    )
    parser.add_argument(
    "--abeta_array", 
    type=int, 
    nargs=2, 
    help="Lista degli stadi Abeta (es: --abeta_array 0 0)"
    )
    parser.add_argument("--st", type=int, help="simulation time")
    parser.add_argument("--best_mod", action="store_true", help="Model parameters optimization to maximize simulation similarity with data")
    parser.add_argument("--fact", type=float, help="diminutive factor in a values")
    parser.add_argument(
        "--impaired_regions",
        nargs="+",               # accepts one or more values
        type=int,                # convert them to integers
        help="List of impaired region indices"
    )

################

################
    args = parser.parse_args()
    gv.show_plots = args.show_plots
    gv.save_plots = args.save_plots
    gv.save_raw_data = args.save_raw_data
    gv.eeg = args.EEG
    gv.psd = args.PSD
    gv.fc = args.FC
    gv.brain_activity = args.brain_act
    gv.skin_and_sensors = args.skin_and_sens
    gv.sim_vs_real = args.sim_vs_real
    gv.connectivity_simnibs = args.conn_sim

    if args.g:
        gv.coup = args.g

    if args.rnd:
        gv.random_seed = args.rnd

    if args.fact:
        gv.factor = args.fact

    if args.st:
        gv.sim_time = args.st
 
    if args.vel:
        gv.vel = args.vel
    
    if args.noise:
        gv.noi = args.noise
    
    if args.a:
        gv.aa = args.a
    
    if args.b:
        gv.bb = args.b

    if args.th:
        gv.th = args.th
    else:
        gv.th = 0.08

    if args.tau:
        gv.tau_stage = args.tau
    else:
        if args.tau_array:
            gv.tau_stage = args.tau_array
        else:
            gv.tau_stage = 0
    
    if args.abeta:
        gv.abeta_stage = args.abeta
    else:
        if args.abeta_array:
            gv.abeta_stage = args.abeta_array
        else:
            gv.abeta_stage = 0

    if args.temp_avg_period:
        gv.temp_avg_period = args.temp_avg_period

    if args.save_folder:
        gv.fig_folder = os.path.join(gv.fig_folder, args.save_folder)
        gv.data_folder = os.path.join(gv.data_folder, args.save_folder)
        if len(gv.fig_folder) > 160:
            raise AttributeError(
                "file directory has to be shorter."
            )

    if args.save_plots:
        os.makedirs(gv.data_folder, exist_ok=True)
        os.makedirs(gv.fig_folder, exist_ok=True)

    if args.struct_conn:
        gv.struct_path = args.struct_conn

    if args.impaired_regions:
        gv.impaired_regions = args.impaired_regions

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
    x_conn_76, y_conn_76, z_conn_76 = conn_76.centres.T[:, :]

    conn_86 = create_connectivity86()
    x_conn_86, y_conn_86, z_conn_86 = conn_86.centres.T[:, :]

    x_conn_76_width = np.abs(np.max(x_conn_76)-np.min(x_conn_76))
    y_conn_76_width = np.abs(np.max(y_conn_76)-np.min(y_conn_76))
    z_conn_76_width = np.abs(np.max(z_conn_76)-np.min(z_conn_76))
    x_conn_76_center = (np.max(x_conn_76)+np.min(x_conn_76))/2
    y_conn_76_center = (np.max(y_conn_76)+np.min(y_conn_76))/2
    z_conn_76_center = (np.max(z_conn_76)+np.min(z_conn_76))/2
    x_conn_86_width = np.abs(np.max(x_conn_86)-np.min(x_conn_86))
    y_conn_86_width = np.abs(np.max(y_conn_86)-np.min(y_conn_86))
    z_conn_86_width = np.abs(np.max(z_conn_86)-np.min(z_conn_86))
    x_conn_86_center = (np.max(x_conn_86)+np.min(x_conn_86))/2
    y_conn_86_center = (np.max(y_conn_86)+np.min(y_conn_86))/2
    z_conn_86_center = (np.max(z_conn_86)+np.min(z_conn_86))/2

    # trasformation of coordinates to make the two boxes coincides
    # in particolare, siccome voglio mappare le regioni della connettivity 76 su quelle della connectivity 86, e siccome non voglio modificare le coordinate dei vertici,
    # mi conviene spostare il cubo_76 sul cubo_86

    ########## PROBLEMA: l'asse z coincide, ma gli assi x e y no. Ruotiamo il sistema attorno all'asse z di pi/2

    x_conn_76_rel = x_conn_86_center + (x_conn_86_width / y_conn_76_width) * (y_conn_76 - y_conn_76_center)
    y_conn_76_rel = y_conn_86_center - (y_conn_86_width / x_conn_76_width) * (x_conn_76 - x_conn_76_center)
    z_conn_76_rel = z_conn_86_center + (z_conn_86_width / z_conn_76_width) * (z_conn_76 - z_conn_76_center)
    x_conn_76_center_rel = (np.max(x_conn_76_rel)+np.min(x_conn_76_rel))/2
    y_conn_76_center_rel = (np.max(y_conn_76_rel)+np.min(y_conn_76_rel))/2
    z_conn_76_center_rel = (np.max(z_conn_76_rel)+np.min(z_conn_76_rel))/2

    centres_76_rel = np.column_stack([x_conn_76_rel, y_conn_76_rel, z_conn_76_rel])

    from sklearn.decomposition import PCA

    pca_76 = PCA(n_components=3)
    pca_86 = PCA(n_components=3)

    pca_76.fit(conn_76.centres)
    pca_86.fit(conn_86.centres)

    print("Assi principali conn_76:\n", pca_76.components_)
    print("Assi principali conn_86:\n", pca_86.components_)

    # La rotazione tra i due è:
    R = pca_76.components_.T @ pca_86.components_
    print("Matrice di rotazione stimata:\n", R)


    # Applica la rotazione PCA
    centres_76_pca_aligned = (R @ conn_76.centres.T).T

    # Riscala e trasla sul centro di conn_86
    centre_86 = conn_86.centres.mean(axis=0)
    centre_76 = centres_76_pca_aligned.mean(axis=0)
    centres_76_pca_aligned = centres_76_pca_aligned - centre_76 + centre_86

    # Visualizza
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*conn_76.centres.T, c='b', label='conn_76 originale', alpha=0.3)
    ax.scatter(*centres_76_pca_aligned.T, c='g', label='conn_76 allineata PCA')
    ax.scatter(*conn_86.centres.T, c='r', label='conn_86')
    ax.legend()

    # Visualizza
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*centres_76_pca_aligned.T, c='g', label='conn_76 allineata PCA')
    ax.scatter(*conn_86.centres.T, c='r', label='conn_86')
    ax.legend()
    plt.close()

    # Scala isotropica basata sulla deviazione standard dei punti
    scale_76 = conn_76.centres.std()
    scale_86 = conn_86.centres.std()
    scale_factor = scale_86 / scale_76

    print(f"Scale factor: {scale_factor:.3f}")

    # Applica rotazione + scala + traslazione
    centres_76_aligned = (R @ conn_76.centres.T).T  # rotazione
    centres_76_aligned = centres_76_aligned * scale_factor  # scala
    centre_86 = conn_86.centres.mean(axis=0)
    centre_76 = centres_76_aligned.mean(axis=0)
    centres_76_aligned = centres_76_aligned - centre_76 + centre_86  # traslazione

    # Visualizza
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(*conn_76.centres.T, c='b', label='conn_76 originale', alpha=0.3)
    ax.scatter(*centres_76_aligned.T, c='g', label='conn_76 allineata e riscalata')
    ax.scatter(*conn_86.centres.T, c='r', label='conn_86')
    ax.legend()

    # Visualizza
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(*conn_76.centres.T, c='b', label='conn_76 originale', alpha=0.3)
    #ax.scatter(*centres_76_aligned.T, c='g', label='conn_76 allineata e riscalata')
    ax.scatter(*conn_86.centres.T, c='r', label='conn_86')
    ax.legend()

    # Visualizza
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(*conn_76.centres.T, c='b', label='conn_76 originale', alpha=0.3)
    ax.scatter(*centres_76_aligned.T, c='g', label='conn_76 allineata e riscalata')
    #ax.scatter(*conn_86.centres.T, c='r', label='conn_86')
    ax.legend()
    plt.show()

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_conn_76_center, y_conn_76_center, z_conn_76_center, color='g', label='Center_76')
    ax.scatter(x_conn_76_center_rel, y_conn_76_center_rel, z_conn_76_center_rel, color='b', label='Center_76_trsnsformed', alpha=0.5)
    ax.scatter(x_conn_86_center, y_conn_86_center, z_conn_86_center, color='r', label='Center_86', alpha=0.5)
    plt.legend()
    plt.show()
    


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_conn_76, y_conn_76, z_conn_76, color='b')
    ax.set_title('Connectivity_76')
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(x_conn_86, y_conn_86, z_conn_86, color='r')
    ax2.set_title('Connectivity_86')

    fig3 = plt.figure()
    ax = fig3.add_subplot(111, projection='3d')

    ax.scatter(x_conn_76_rel, y_conn_76_rel, z_conn_76_rel, color='b', label='Connectivity_76')
    ax.scatter(x_conn_86, y_conn_86, z_conn_86, color='r', label='Connectivity_86')
    ax.set_title('Connectivities comparison')
    plt.legend()
    plt.show()
    """
    regions_76_abeta_stages=np.zeros(76)

    for i in Abeta_stage_1:
        regions_76_abeta_stages[i]=3
    for i in Abeta_stage_2:
        regions_76_abeta_stages[i]=2
    for i in Abeta_stage_3:
        regions_76_abeta_stages[i]=1

    regions_76_tau_stages=np.zeros(76)

    for i in tau_stage_1:
        regions_76_tau_stages[i]=3
    for i in tau_stage_2:
        regions_76_tau_stages[i]=2
    for i in tau_stage_3:
        regions_76_tau_stages[i]=1

    print('Suddivisione regioni nella parcellizzazione a 76 regioni')
    print('Abeta protein')
    print(regions_76_abeta_stages)
    print('Tau_protein')
    print(regions_76_tau_stages)
    
    ax = plt.subplot(111, projection='3d')
    tree = KDTree(conn_76.centres)
    cortex = surfaces.CorticalSurface.from_file(r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\sub-CLB00044_Cortex.zip")
    
    cortex.configure()
    _, nearest_region = tree.query(cortex.vertices)

    # Mappa i valori delle regioni sui vertici
    vertex_values_abeta = regions_76_abeta_stages[nearest_region]
    vertex_values_tau = regions_76_tau_stages[nearest_region]
            
    # Calcola il colore medio per ogni triangolo
    tri_values_abeta = vertex_values_abeta[cortex.triangles].mean(axis=1)
    """
    norm = plt.Normalize(vmin=regions_76_abeta_stages.min(), vmax=regions_76_abeta_stages.max())
    cmap = plt.cm.hot
    face_colors = cmap(norm(tri_values))
    # Costruisci i triangoli come lista di coordinate
    verts = cortex.vertices[cortex.triangles]  # shape (M, 3, 3)
    
    poly = Poly3DCollection(verts, alpha=0.8, edgecolor='none')
    poly.set_facecolor(face_colors)
    ax.add_collection3d(poly)

    # Aggiorna i limiti degli assi manualmente
    x_cortex, y_cortex, z_cortex = cortex.vertices.T
    ax.set_xlim(x_cortex.min(), x_cortex.max())
    ax.set_ylim(y_cortex.min(), y_cortex.max())
    ax.set_zlim(z_cortex.min(), z_cortex.max())
    sc = ax.scatter(x_conn_76_rel, y_conn_76_rel, z_conn_76_rel, c=regions_76_abeta_stages, cmap='hot', s=50, zorder=5)
    plt.colorbar(sc, ax=ax, shrink=0.5, label='Value')
    """
    region_mapping = np.loadtxt(r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\sub-CLB00044_region_mapping.txt")

    from scipy import stats

    ###### ABETA ######

    regions_86_abeta_stages = np.zeros(86, dtype=int)

    for region in range(86):
        # Trova i vertici appartenenti a questa regione
        mask = region_mapping == region
        if mask.sum() > 0:
            # Prendi il valore più frequente (moda)
            regions_86_abeta_stages[region] = stats.mode(vertex_values_abeta[mask], keepdims=True).mode[0]
    
    print('Suddivisione 86 regioni con region mapping: proteina abeta')
    print(regions_86_abeta_stages)

    tree = KDTree(conn_86.centres)
    _, nearest_region = tree.query(cortex.vertices)

    regions_86_abeta_stages = np.zeros(86, dtype=int)

    for region in range(86):
        mask = nearest_region == region
        if mask.sum() > 0:
            regions_86_abeta_stages[region] = stats.mode(vertex_values_abeta[mask], keepdims=True).mode[0]

    print('Suddivisione 86 regioni con nearest region: proteina abeta')
    print(regions_86_abeta_stages)

    centres_76_rel = np.column_stack([x_conn_76_rel, y_conn_76_rel, z_conn_76_rel])
    tree = KDTree(centres_76_rel)
    _, nearest_region_76 = tree.query(conn_86.centres)

    regions_86_abeta_stages = regions_76_abeta_stages[nearest_region_76]

    print('Suddivisione 86 regioni con KDTree sui centri: proteina abeta')
    print(regions_86_abeta_stages)

    tree = KDTree(conn_76.centres)
    _, nearest_region = tree.query(cortex.vertices)

    regions_76_abeta_stages = np.zeros(76, dtype=int)

    for region in range(76):
        mask = nearest_region == region
        if mask.sum() > 0:
            regions_76_abeta_stages[region] = stats.mode(vertex_values_abeta[mask], keepdims=True).mode[0]

    print('Suddivisione 76 regioni con nearest region: proteina abeta')
    print(regions_76_abeta_stages)

    ###### TAU ######
    
    regions_86_tau_stages = np.zeros(86, dtype=int)

    for region in range(86):
        # Trova i vertici appartenenti a questa regione
        mask = region_mapping == region
        if mask.sum() > 0:
            # Prendi il valore più frequente (moda)
            regions_86_tau_stages[region] = stats.mode(vertex_values_tau[mask], keepdims=True).mode[0]

    print('Suddivisione 86 regioni con region mapping: proteina tau')
    print(regions_86_tau_stages)

    tree = KDTree(conn_86.centres)
    _, nearest_region = tree.query(cortex.vertices)

    regions_86_tau_stages = np.zeros(86, dtype=int)

    for region in range(86):
        mask = nearest_region == region
        if mask.sum() > 0:
            regions_86_tau_stages[region] = stats.mode(vertex_values_tau[mask], keepdims=True).mode[0]

    print('Suddivisione 86 regioni con nearest region: proteina tau')
    print(regions_86_tau_stages)

    tree = KDTree(centres_76_rel)
    _, nearest_region_76 = tree.query(conn_86.centres)

    regions_86_tau_stages = regions_76_tau_stages[nearest_region_76]

    print('Suddivisione 86 regioni con KDTree sui centri: proteina tau')
    print(regions_86_tau_stages)
    
    #plt.close()

    if gv.skin_and_sensors == True:

        #skin_and_sensors()
            # Plot cortical regions colorate per valore
        skin = surfaces.SkinAir.from_file(r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\sub-CLB00044_outer_skin.zip")
        skin.configure()
        cortex = surfaces.CorticalSurface.from_file(r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\sub-CLB00044_Cortex.zip")
        cortex.configure()
        x_cortex, y_cortex, z_cortex = cortex.vertices.T
        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, projection='3d')
        conn = gv.get_conn()
        x_conn, y_conn, z_conn = conn.centres.T[:, :]
        
        region_values = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\stimulus_vector_TVB.npy")
        
        """region_values = np.array([2., 2., 3., 3., 1., 2., 3., 2., 2., 2., 2., 2., 2., 2., 1., 3., 2.,
       2., 2., 2., 2., 1., 3., 3., 2., 2., 3., 3., 2., 2., 2., 1., 2., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 2.,
       3., 3., 1., 2., 3., 2., 2., 2., 2., 2., 2., 2., 1., 3., 2., 2., 2.,
       2., 3., 2., 3., 3., 2., 2., 3., 3., 2., 2., 2., 1., 2., 1., 1., 1.,
       1.])"""
        region_values_76 = np.zeros((76,1))

        for i in range(len(region_values_76)):
            if i < 20:
                region_values_76[i] = 1
            if 20 <= i < 40:
                region_values_76[i] = 2
            if i >= 40:
                region_values_76[i] = 3
        #region_values = region_values_76
        if region_values is not None:
            sc = ax.scatter(x_conn, y_conn, z_conn, c=region_values, cmap='hot', s=50, zorder=5)
            plt.colorbar(sc, ax=ax, shrink=0.5, label='Valore')
        else:
            ax.plot(x_conn, y_conn, z_conn, 'ko')
        
        ax.plot(x_conn, y_conn, z_conn, 'ko')
        #ax.plot_trisurf(x_cortex, y_cortex, z_cortex, triangles=cortex.triangles, alpha=0.1, edgecolor='k')

            # Plot corteccia colorata per regione
        
        if region_values is not None:
            # Per ogni vertice, trova la regione più vicina
            tree = KDTree(conn.centres)
            _, nearest_region = tree.query(cortex.vertices)

            # Mappa i valori delle regioni sui vertici
            vertex_values = region_values[nearest_region]
            
            # Calcola il colore medio per ogni triangolo
            tri_values = vertex_values[cortex.triangles].mean(axis=1)
            norm = plt.Normalize(vmin=region_values.min(), vmax=region_values.max())
            cmap = plt.cm.hot
            face_colors = cmap(norm(tri_values))
            # Costruisci i triangoli come lista di coordinate
            verts = cortex.vertices[cortex.triangles]  # shape (M, 3, 3)

            poly = Poly3DCollection(verts, alpha=0.8, edgecolor='none')
            poly.set_facecolor(face_colors)
            ax.add_collection3d(poly)

            # Aggiorna i limiti degli assi manualmente
            ax.set_xlim(x_cortex.min(), x_cortex.max())
            ax.set_ylim(y_cortex.min(), y_cortex.max())
            ax.set_zlim(z_cortex.min(), z_cortex.max())
        else:
            ax.plot_trisurf(x_cortex, y_cortex, z_cortex, triangles=cortex.triangles,
                            alpha=0.1, edgecolor='k')
        
        #x, y, z = skin.vertices.T
        #ax.plot_trisurf(x, y, z, triangles=skin.triangles, alpha=0.1, edgecolor='k')
        plt.show()

if __name__ == "__main__":
    main()
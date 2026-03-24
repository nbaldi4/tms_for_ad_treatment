import os

import numpy as np
from tvb.simulator.lab import *
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import stats

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

    regions_76_abeta_stages=np.zeros(76)
    v1_76_vector = np.zeros(76)
    v1_76_vector[35] = 1
    v1_76_vector[73] = 1

    precunesu_86_vector = np.zeros(86)
    precunesu_86_vector[24] = 1
    precunesu_86_vector[73] = 1

    frontal_pole_86_vector = np.zeros(86)
    frontal_pole_86_vector[31] = 1
    frontal_pole_86_vector[80] = 1

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
    conn_86 = create_connectivity86()

    def realign_conn(conn_76, conn_86):
        x_conn_76, y_conn_76, z_conn_76 = conn_76.centres.T[:, :]
        x_conn_86, y_conn_86, z_conn_86 = conn_86.centres.T[:, :]

        # questo passaggio serve per allineare i centri della connettività 76 con i centri della connettività 86.
        # Infatti la connettività 86 risulta ruotata e traslata rispetto alla connettività 76.
        # Dovendo però utilizzare la corteccia 86, risulta più conventiente spostare i centri 76 per allinearli ai centri 86 piuttosto che il contrario.
    
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
        #x_conn_76_rel = x_conn_86_center - (x_conn_86_width / y_conn_76_width) * (y_conn_76 - y_conn_76_center)
        #y_conn_76_rel = y_conn_86_center + (y_conn_86_width / x_conn_76_width) * (x_conn_76 - x_conn_76_center)
        #z_conn_76_rel = z_conn_86_center + (z_conn_86_width / z_conn_76_width) * (z_conn_76 - z_conn_76_center)
        x_conn_76_center_rel = (np.max(x_conn_76_rel)+np.min(x_conn_76_rel))/2
        y_conn_76_center_rel = (np.max(y_conn_76_rel)+np.min(y_conn_76_rel))/2
        z_conn_76_center_rel = (np.max(z_conn_76_rel)+np.min(z_conn_76_rel))/2

        return x_conn_76_rel, y_conn_76_rel, z_conn_76_rel
    
    x_conn_76_rel, y_conn_76_rel, z_conn_76_rel = realign_conn(conn_76=conn_76, conn_86=conn_86)

    centres_76_rel = np.column_stack([x_conn_76_rel, y_conn_76_rel, z_conn_76_rel])

    print('Suddivisione regioni nella parcellizzazione a 76 regioni')
    print('Abeta protein')
    print(regions_76_abeta_stages)
    print('Tau_protein')
    print(regions_76_tau_stages)

    region_mapping = np.loadtxt(r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\sub-CLB00044_region_mapping.txt")

    def region_mapping_staging(region_76_staging, title, region_mapping, centres=centres_76_rel):

        tree = KDTree(centres)
        cortex = surfaces.CorticalSurface.from_file(r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\sub-CLB00044_Cortex.zip")
        
        cortex.configure()
        _, nearest_region = tree.query(cortex.vertices)

        # Mappa i valori delle regioni sui vertici
        vertex_values = region_76_staging[nearest_region]

        regions_86_stages = np.zeros(86, dtype=int)

        stages_0_counter = 0
        stages_1_counter = 0
        stages_2_counter = 0
        stages_3_counter = 0

        for region in range(86):
            # Trova i vertici appartenenti a questa regione
            mask = region_mapping == region
            if mask.sum() > 0:
                # Prendi il valore più frequente (moda)
                regions_86_stages[region] = stats.mode(vertex_values[mask], keepdims=True).mode[0]

        for region in range(86):
            if regions_86_stages[region] == 0: stages_0_counter += 1
            if regions_86_stages[region] == 1: stages_1_counter += 1
            if regions_86_stages[region] == 2: stages_2_counter += 1
            if regions_86_stages[region] == 3: stages_3_counter += 1
    
        print(title)
        print('number of regions in stage 0: ', stages_0_counter)
        print('number of regions in stage 1: ', stages_1_counter)
        print('number of regions in stage 2: ', stages_2_counter)
        print('number of regions in stage 3: ', stages_3_counter)
        print(regions_86_stages)

        return regions_86_stages

    def nearest_vertices_staging(region_76_staging, title, centres=centres_76_rel):

        cortex = surfaces.CorticalSurface.from_file(r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\sub-CLB00044_Cortex.zip")
        cortex.configure()

        # 1. Assegna ogni vertice alla regione 76 più vicina
        tree_76 = KDTree(centres)
        _, nearest_region_76 = tree_76.query(cortex.vertices)

        # 2. Assegna il valore dello staging 76 a ogni vertice
        vertex_values = region_76_staging[nearest_region_76]

        # 3. Assegna ogni vertice alla regione 86 più vicina
        tree_86 = KDTree(conn_86.centres)
        _, nearest_region_86 = tree_86.query(cortex.vertices)

        # 4. Per ogni regione 86, voto a maggioranza sui vertici assegnati
        regions_86_stages = np.zeros(86, dtype=int)
        for region in range(86):
            mask = nearest_region_86 == region
            if mask.sum() > 0:
                regions_86_stages[region] = stats.mode(vertex_values[mask], keepdims=True).mode[0]

        # Contatori
        stages_0_counter = np.sum(regions_86_stages == 0)
        stages_1_counter = np.sum(regions_86_stages == 1)
        stages_2_counter = np.sum(regions_86_stages == 2)
        stages_3_counter = np.sum(regions_86_stages == 3)

        print(title)
        print('number of regions in stage 0: ', stages_0_counter)
        print('number of regions in stage 1: ', stages_1_counter)
        print('number of regions in stage 2: ', stages_2_counter)
        print('number of regions in stage 3: ', stages_3_counter)
        print(regions_86_stages)

        return regions_86_stages

    def nearest_center_staging(region_76_staging, title, centres=centres_76_rel):

        tree = KDTree(centres)
        _, nearest_region_76 = tree.query(conn_86.centres)

        regions_86_stages = region_76_staging[nearest_region_76]

        stages_0_counter = 0
        stages_1_counter = 0
        stages_2_counter = 0
        stages_3_counter = 0

        stage_0_regions = []
        stage_1_regions = []
        stage_2_regions = []
        stage_3_regions = []

        for region in range(86):
            if regions_86_stages[region] == 0: 
                stages_0_counter += 1
                stage_0_regions.append(region)
            if regions_86_stages[region] == 1: 
                stages_1_counter += 1
                stage_1_regions.append(region)
            if regions_86_stages[region] == 2: 
                stages_2_counter += 1
                stage_2_regions.append(region)
            if regions_86_stages[region] == 3: 
                stages_3_counter += 1
                stage_3_regions.append(region)     
        
        print(title)
        print('number of regions in stage 0: ', stages_0_counter)
        print(stage_0_regions)
        print('number of regions in stage 1: ', stages_1_counter)
        print(stage_1_regions)
        print('number of regions in stage 2: ', stages_2_counter)
        print(stage_2_regions)
        print('number of regions in stage 3: ', stages_3_counter)
        print(stage_3_regions)
        print('all regions stages')
        print(regions_86_stages)

        return regions_86_stages

    def plot_regions(centres, region_values, title, cortex, simnibs):
        tree = KDTree(centres)
        cortex.configure()
        if simnibs == True:
            angle = np.radians(90)
            R = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle),  np.cos(angle), 0],
                [0,              0,             1]
            ])
            cortex.vertices = cortex.vertices @ R.T
        
        x_cortex, y_cortex, z_cortex = cortex.vertices.T

        _, nearest_region = tree.query(cortex.vertices)
        vertex_values = region_values[nearest_region]
        tri_values = vertex_values[cortex.triangles].mean(axis=1)
        norm = plt.Normalize(vmin=region_values.min(), vmax=region_values.max())
        cmap = plt.cm.hot
        face_colors = cmap(norm(tri_values))
        verts = cortex.vertices[cortex.triangles]

        views = [
            ('Top',      90,   -90),
            ('Bottom',  -90,   -90),
            ('Frontal',   0,     0),
            ('Back',      0,   180),
            ('Lateral sx',0,   -90),
            ('Lateral dx',0,    90),
        ]

        fig = plt.figure(figsize=(18, 12))

        for i, (view_name, elev, azim) in enumerate(views):
            ax = fig.add_subplot(2, 3, i + 1, projection='3d')

            poly = Poly3DCollection(verts, alpha=0.8, edgecolor='none')
            poly.set_facecolor(face_colors)
            ax.add_collection3d(poly)

            ax.set_xlim(x_cortex.min(), x_cortex.max())
            ax.set_ylim(y_cortex.min(), y_cortex.max())
            ax.set_zlim(z_cortex.min(), z_cortex.max())
            ax.set_title(view_name)
            ax.view_init(elev=elev, azim=azim)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=fig.get_axes(), shrink=0.5, pad=0.05, label='stages')

        fig.suptitle(title)
        plt.savefig(os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop", title))
        plt.close()

    def plot_centres(centres, centres_2, region_values_76, regions_values_86, title):
        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, projection='3d')

        x, y, z = centres.T
        sc = ax.scatter(x, y, z, c=region_values_76, cmap='viridis', s=50, marker='o', label='conn_76')
        x2,y2,z2= centres_2.T
        sc = ax.scatter(x2, y2, z2, c=regions_values_86, cmap='viridis', s=50, marker='^', label='conn_86')
        plt.colorbar(sc, ax=ax, shrink=0.5, label='Stage')
        plt.legend()

        ax.set_title(title)
        plt.savefig(os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop\Immagini da far vedere a Lorenzo_20_03", title))
        plt.close()
    
    def plot_regions_2(centres, region_values, title):
        tree = KDTree(centres)
        cortex = surfaces.CorticalSurface.from_file()
        cortex.configure()            
        angle = np.radians(-90)
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,              0,             1]
        ])
        cortex.vertices = cortex.vertices @ R.T
        x_cortex, y_cortex, z_cortex = cortex.vertices.T

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

        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, projection='3d')
        ax.add_collection3d(poly)

        # Aggiorna i limiti degli assi manualmente
        ax.set_xlim(x_cortex.min(), x_cortex.max())
        ax.set_ylim(y_cortex.min(), y_cortex.max())
        ax.set_zlim(z_cortex.min(), z_cortex.max())
        ax.set_title(title)
        
        #x, y, z = skin.vertices.T
        #ax.plot_trisurf(x, y, z, triangles=skin.triangles, alpha=0.1, edgecolor='k')
        plt.savefig(os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop",title))
        plt.close()
    
    region_mapping_abeta_values = region_mapping_staging(centres=centres_76_rel, region_76_staging=regions_76_abeta_stages, title='Suddivisione 86 regioni con region mapping: proteina abeta', region_mapping=region_mapping)
    region_mapping_tau_values = region_mapping_staging(centres=centres_76_rel, region_76_staging=regions_76_tau_stages, title='Suddivisione 86 regioni con region mapping: proteina tau', region_mapping=region_mapping)
    
    nearest_vertices_abeta_values = nearest_vertices_staging(centres=centres_76_rel, region_76_staging=regions_76_abeta_stages, title='Suddivisione 86 regioni con nearest vertices: proteina abeta')
    nearest_vertices_tau_values = nearest_vertices_staging(centres=centres_76_rel, region_76_staging=regions_76_tau_stages, title='Suddivisione 86 regioni con nearest vertices: proteina tau')

    nearest_centre_abeta_values = nearest_center_staging(centres=centres_76_rel, region_76_staging=regions_76_abeta_stages, title='Suddivisione 86 regioni con KDTree sui centri: proteina abeta')
    nearest_centre_tau_values = nearest_center_staging(centres=centres_76_rel, region_76_staging=regions_76_tau_stages, title='Suddivisione 86 regioni con KDTree sui centri: proteina tau')
    cortex_simnibs = r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\sub-CLB00044_Cortex.zip"
    #plot_regions(centres=conn_86.centres, region_values=region_mapping_abeta_values, simnibs=True, title='abeta_staging_connectivity_86_cortex_simnibs_region_mapping', cortex=surfaces.CorticalSurface.from_file(cortex_simnibs))
    #plot_regions(centres=conn_86.centres, region_values=nearest_vertices_abeta_values, simnibs=True, title='abeta_staging_connectivity_86_cortex_simnibs_nearest vertices', cortex=surfaces.CorticalSurface.from_file(cortex_simnibs))
    #plot_regions(centres=conn_86.centres, region_values=frontal_pole_86_vector, simnibs=True, title='frontal_pole_connectivity_86_cortex_simnibs', cortex=surfaces.CorticalSurface.from_file(cortex_simnibs))
    #plot_regions_2(centres=conn_76.centres, region_values=regions_76_abeta_stages, title='abeta_staging_connectivity_76_cortex_tvb_one_view_2')

    """
    plot_regions(centres=centres_76_rel, region_values=regions_76_abeta_stages, title='stages_76_abeta')
    plot_regions(centres=centres_76_rel, region_values=regions_76_tau_stages, title='stages_76_tau')
    plot_regions(centres=conn_86.centres, region_values=region_mapping_abeta_values, title='stages_86_region_mapping_abeta_before_realignment')
    plot_regions(centres=conn_86.centres, region_values=region_mapping_tau_values, title='stages_86_region_mapping_tau_before_realignment')
    plot_regions(centres=conn_86.centres, region_values=nearest_vertices_abeta_values, title='stages_86_nearest_vertices_abeta_before_realignment')
    plot_regions(centres=conn_86.centres, region_values=nearest_vertices_tau_values, title='stages_86_nearest_vertices_tau_before_realignment')
    plot_regions(centres=conn_86.centres, region_values=nearest_centre_abeta_values, title='stages_86_nearest_centre_abeta_before_realignment')
    plot_regions(centres=conn_86.centres, region_values=nearest_centre_tau_values, title='stages_86_nearest_centre_tau_before_realignment')
    

    plot_centres(centres=conn_76.centres, centres_2=conn_86.centres, region_values_76=regions_76_abeta_stages, regions_values_86=region_mapping_abeta_values, title='before_realignment')
    plot_centres(centres=centres_76_rel, centres_2=conn_86.centres, region_values_76=regions_76_abeta_stages, regions_values_86=region_mapping_abeta_values, title='after_realignment')
    """

if __name__ == "__main__":
    main()
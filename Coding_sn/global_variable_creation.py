from tvb.simulator.lab import monitors, connectivity
import numpy as np
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.projections import  ProjectionSurfaceEEG#,ProjectionMatrix,
from tvb.datatypes.sensors import SensorsEEG

show_plots = False
real_data = False
save_plots = False
sim_vs_real = False
th = 0.08
tau_stage = 0
random_seed = 0
abeta_stage = 0
temp_avg_period = 1.0 #ms
fig_folder = r'..\Results_figures'
data_folder = r'..\Results_data'
struct_path = r'..\Degeneration_data\tau_matrices.npy'
struct_path_connectivity_86 = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Degeneration_data\tau_matrices_connectivity_86.npy"
a_path = r'..\Degeneration_data\a_stages.npy'
b_path = r'..\Degeneration_data\b_stages.npy'
HC_FC_data = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\HC_data\fc_ctr_preview.npy"
HC_PSD_data = r'..\HC_data\psd_ctr_preview.npy'
MCI_FC_data = r'..\MCI_data\fc_mci_preview.npy'
MCI_PSD_data = r'..\MCI_data\psd_mci_preview.npy'
impaired_regions = [21,22,30,31,32,34,59,60,68,69,70,72]
partitioning = 6
eeg = True
psd = True
save_raw_data = False
fc = True
stim = True
sim_time = 20000
brain_activity = True
skin_and_sensors = True
coup=5.217348458621988#0.10333870704798209#0.10004127233208432#0.10245373952358368#
vel=64.21898968502352#26.760911328133584#10.374658598387436#41.79385759129675#
noi=0.040677673235302345#0.1077485312389941#0.14480848489827142#0.10045650478858126#
aa=0.07125#0.07406834111496974#0.06625#0.06897011085340124#                    0.0728044836955774#0.09806515133627744#
bb=0.0546595396476822#0.06602933308057984#0.06293568322880852#0.0680013974341444#
factor = -0.12#0#      

ref_PT=np.array([1.05, 1.05, 0.55])#ref_PT=np.array([2.0, 2.0, 30.0])
connectivity_simnibs = True

def eeg_monitor_factory():
    """Factory che gestisce un monitor EEG con configurazione persistente."""
    if connectivity_simnibs == True:
        """Crea un monitor EEG, ricreandolo ad ogni chiamata perché dipende da conn."""
        SENSORS_FNAME = r'C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\sub-CLB00044_EEG_Locations.txt'
        PROJECTION_FNAME = r'C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\sub-CLB00044_EEGProjection.mat'
        RM_FNAME = r'C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\sub-CLB00044_region_mapping.txt'

        # Carica una volta sola i file statici
        sensorsEEG = SensorsEEG.from_file(SENSORS_FNAME)
        sensorsEEG.configure()

        def create_eeg_monitor(conn=get_conn(), period=temp_avg_period):
            # rm dipende da conn -> ricreato ad ogni chiamata
            rm = RegionMapping.from_file(RM_FNAME)
            rm.connectivity = conn

            prEEG = ProjectionSurfaceEEG.from_file(
                PROJECTION_FNAME,
                matlab_data_name='ProjectionMatrix'
            )
            prEEG.sensors = sensorsEEG

            mon_eeg = monitors.EEG(
                sensors=sensorsEEG,
                projection=prEEG,
                region_mapping=rm,
                period=period,
                reference=None,
                variables_of_interest=np.array([0])
            )
            mon_eeg.configure()
            return mon_eeg
    else:
        config = {
            'sensors_fname': 'eeg_brainstorm_65.txt',
            'projection_fname': 'projection_eeg_65_surface_16k.npy',
            'rm_f_name': 'regionMapping_16k_76.txt',
            'period': temp_avg_period
        }
        cached_mon = None
        cached_conf = config.copy()

        def create_eeg_monitor(**kwargs):
            nonlocal config, cached_mon, cached_conf

            # Aggiorna la configurazione se arrivano nuovi parametri
            if kwargs:
                config.update(kwargs)

            # Se la configurazione non è cambiata, usa il monitor in cache
            if cached_mon and config == cached_conf:
                return cached_mon

            # Altrimenti ricrea il monitor
            mon_EEG = monitors.EEG.from_file(**config)
            mon_EEG.configure()

            cached_mon = mon_EEG
            cached_conf = config.copy()
            return mon_EEG

    return create_eeg_monitor

def connectivity_factory():
    if connectivity_simnibs == True:
        """Factory che gestisce un oggetto Connectivity con configurazione persistente."""
        filename = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\HC_data\real_patient_connectome.zip"
        config = {
            'filename': filename
        }
        cached_conn = None
        cached_conf = config.copy()

        def create_connectivity(**kwargs):
            nonlocal config, cached_conn, cached_conf

            # Aggiorna la configurazione se arrivano nuovi parametri
            if kwargs:
                config.update(kwargs)

            # Se la configurazione non è cambiata, usa la connessione in cache
            if cached_conn and config == cached_conf:
                return cached_conn

            # Altrimenti crea una nuova Connectivity
            conn = connectivity.Connectivity.from_file(config['filename'])
            conn.configure()  # opzionale, se serve come per EEG

            cached_conn = conn
            cached_conf = config.copy()
            return conn
    else:
        """Factory che gestisce un oggetto Connectivity con configurazione persistente."""
        config = {
            'filename': 'connectivity_76.zip'
        }
        cached_conn = None
        cached_conf = config.copy()

        def create_connectivity(**kwargs):
            nonlocal config, cached_conn, cached_conf

            # Aggiorna la configurazione se arrivano nuovi parametri
            if kwargs:
                config.update(kwargs)

            # Se la configurazione non è cambiata, usa la connessione in cache
            if cached_conn and config == cached_conf:
                return cached_conn

            # Altrimenti crea una nuova Connectivity
            conn = connectivity.Connectivity.from_file(config['filename'])
            conn.configure()  # opzionale, se serve come per EEG

            cached_conn = conn
            cached_conf = config.copy()
            return conn

    return create_connectivity

# Istanza globale unica — importabile ovunque
get_conn = connectivity_factory()
get_eeg_monitor = eeg_monitor_factory()

#mon_eeg = get_eeg_monitor() # Questo chiama la tua factory
#print("Nomi sensori EEG simulati:")
#print(mon_eeg.sensors.labels)
"""
['Fp1' 'Fp2' 'F4' 'F3' 'C3' 'C4' 'P4' 'P3' 'O2' 'O1' 'F8' 'F7' 'T8/T4'
 'T7/T3' 'P8/T6' 'P7/T5' 'Pz' 'Fz' 'IO1' 'IO2' 'AF9' 'AF10' 'F9' 'F10'
 'CB1' 'CB2' 'TP7' 'TP9' 'TP10' 'TP8' 'Oz' 'Iz' 'PO4' 'PO3' 'CP5' 'CP6'
 'CP1' 'CP2' 'FT9' 'FT10' 'FC2' 'FC1' 'AF3' 'AF4' 'FC6' 'FC5' 'CPz' 'P1'
 'POz' 'P2' 'P6' 'C6' 'P5' 'C1' 'C2' 'C5' 'F2' 'F6' 'F1' 'AF8' 'F5' 'AF7'
 'Fpz' 'FCz' 'Cz']
 """
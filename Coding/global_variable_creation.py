from tvb.simulator.lab import monitors, connectivity
import numpy as np

show_plots = False
real_data = False
save_plots = False
sim_vs_real = False
th = 0.08
tau_stage = 0
abeta_stage = 0
temp_avg_period = 1.0 #ms
fig_folder = r'..\Results_figures'
data_folder = r'..\Results_data'
struct_path = r'..\Degeneration_data\tau_matrices.npy'
a_path = r'..\Degeneration_data\a_stages.npy'
b_path = r'..\Degeneration_data\b_stages.npy'
HC_FC_data = r'..\HC_data\fc_preview_ctr_alpha.npy'
HC_PSD_data = r'..\HC_data\psd_ctr_preview.npy'
MCI_FC_data = r'..\MCI_data\fc_mci_preview.npy'
MCI_PSD_data = r'..\MCI_data\psd_mci_preview.npy'
impaired_regions = [21,22,30,31,32,34,59,60,68,69,70,72]
partitioning = 6
eeg = True
psd = True
fc = True
sim_time = 20000
brain_activity = True
skin_and_sensors = True
coup=0.10004127233208432#0.10245373952358368
vel=10.374658598387436#41.79385759129675
noi=0.14480848489827142#0.10045650478858126
aa=0.0728044836955774#0.09806515133627744
bb=0.06293568322880852#0.0680013974341444
factor = 0
ref_PT=np.array([2.0, 2.0, 30.0])

def eeg_monitor_factory():
    """Factory che gestisce un monitor EEG con configurazione persistente."""
    config = {
        'sensors_fname': 'eeg_brainstorm_65.txt',
        'projection_fname': 'projection_eeg_65_surface_16k.npy',
        'rm_f_name': 'regionMapping_16k_76.txt',
        'period': 1
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
get_eeg_monitor = eeg_monitor_factory()
get_conn = connectivity_factory()

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
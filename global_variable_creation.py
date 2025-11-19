from tvb.simulator.lab import monitors, connectivity
import numpy as np

show_plots = False
real_data = False
save_plots = False
sim_vs_real = False
th = 0.08
temp_avg_period = 1.0 #ms
fig_folder = r'C:\Users\User\OneDrive - University of Pisa\Desktop\test_figures'
struct_path = r'C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\structural_connectivities.npy'
real_FC_data = r'C:\Users\User\OneDrive - University of Pisa\Desktop\TVB_tutorials\Dati_Healthy\fc_preview_ctr_alpha.npy'
impaired_regions = [21,22,30,31,32,34,59,60,68,69,70,72]
partitioning = 6
eeg = True
psd = True
fc = True
brain_activity = True
skin_and_sensors = True

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
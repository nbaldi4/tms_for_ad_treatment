from tvb.simulator.lab import monitors, connectivity

show_plots = True
save_plots = True
th = 0.08
temp_avg_period = 1 #ms
fig_folder = r'C:\Users\User\OneDrive - University of Pisa\Desktop\test_figures'

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

def init_show_plots(**kwargs):
    show_plot_value = kwargs.get('show_plot')
    global show_plots
    show_plots = show_plot_value

def init_save_plots(**kwargs):
    save_plot_value = kwargs.get('show_plot')
    global save_plots
    save_plots = save_plot_value

def init_fc_th(**kwargs):
    th_value = kwargs.get('threshold')
    global th
    th = th_value

def init_temp_avg_period(**kwargs):
    temp_avg_period_value = kwargs.get('period')
    global temp_avg_period
    temp_avg_period = temp_avg_period_value

def init_fig_folder(**kwargs):
    fig_folder_value = kwargs.get('path')
    global fig_folder
    fig_folder = fig_folder_value

# Istanza globale unica — importabile ovunque
get_eeg_monitor = eeg_monitor_factory()
get_conn = connectivity_factory()
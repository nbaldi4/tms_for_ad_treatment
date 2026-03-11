from tvb.simulator.lab import monitors, connectivity
import numpy as np
import scipy.io
import numpy as np
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.projections import  ProjectionSurfaceEEG#,ProjectionMatrix,
from tvb.datatypes.sensors import SensorsEEG
from tvb.simulator.simulator import Simulator as Simulator_fromTVB

class Simulator_Modified(Simulator_fromTVB):

        def _loop_update_stimulus(self, step, stimulus):
            """Update stimulus values for current time step."""
            if self.stimulus is not None:
                # TODO stim_step != current step
                stim_step = step - (self.current_step + 1)
                #stimulus[self.model.stvar, :, :] = self.stimulus(stim_step).reshape((1, -1, 1)) #THIS IS THE ORIGINAL LINE FROM TVB AS IN THE FILE simulator.py (the lines below the current are not present in the original version)
                if self.surface==None:
                    stimulus[self.model.stvar, :, :] = self.stimulus(stim_step).reshape((1, -1, 1))
                if self.surface!=None:
                    stimulus[self.model.stvar, :, :] = np.append(self.stimulus(stim_step), np.zeros((int(self.connectivity.cortical.shape[0]-np.count_nonzero(self.connectivity.cortical)),1))).reshape((1, -1, 1))

        def _configure_stimuli(self):
            """ Configure the defined Stimuli for this Simulator """
            if self.stimulus is not None:
                if self.surface:
                    # NOTE the region mapping of the stimuli should also include the subcortical areas
                    #self.stimulus.configure_space(region_mapping=self.surface.region_mapping)
                    pass
                else:
                    self.stimulus.configure_space()

        def run(self, **kwds):
            """Convenience method to call the simulator with **kwds and collect output data."""
            import time # this line has been added respect to TVB original file, this is needed otherwise returns error as "time" is undefined...
            ts, xs = [], []
            for _ in self.monitors:
                ts.append([])
                xs.append([])
            wall_time_start = time.time()
            for data in self(**kwds):
                for tl, xl, t_x in zip(ts, xs, data):
                    if t_x is not None:
                        t, x = t_x
                        tl.append(t)
                        xl.append(x)
                        if np.any(np.isnan(x)): # this line and the next 4 lines have been added respect to TVB original file: they break the two "for" nested loops when the last output of variable "x" contains one or more "nan" values
                            break
                else:
                    continue
                break
            elapsed_wall_time = time.time() - wall_time_start
            #self.log.info("%.3f s elapsed, %.3fx real time", elapsed_wall_time,
            #              elapsed_wall_time * 1e3 / self.simulation_length)
            self.log.info("")
            for i in range(len(ts)):
                ts[i] = np.array(ts[i])
                xs[i] = np.array(xs[i])
            return list(zip(ts, xs))    
       

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
sim_time = 20000
brain_activity = True
skin_and_sensors = True
coup=0.10333870704798209#0.10004127233208432#0.10245373952358368#
vel=26.760911328133584#10.374658598387436#41.79385759129675#
noi=0.1077485312389941#0.14480848489827142#0.10045650478858126#
aa=0.06625#0.06897011085340124#                    0.0728044836955774#0.09806515133627744#
bb=0.06602933308057984#0.06293568322880852#0.0680013974341444#
factor = -0.08#0#
ref_PT=np.array([2.0, 2.0, 30.0])

def real_eeg_monitor_factory(real_conn):
    """Factory che gestisce un monitor EEG con configurazione persistente."""
    real_config = {
        'sensors_fname': r'C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\sub-CLB00044_EEG_Locations.txt',
        'projection_fname': r'C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\sub-CLB00044_EEGProjection.mat',
        'projection_matlab_name': 'ProjectionMatrix',
        'rm_fname': r'C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\sub-CLB00044_region_mapping.txt',
        'period': temp_avg_period
    }
    cached_real_mons = None
    cached_real_conf = real_config.copy()

    def create_real_eeg_monitor(**kwargs):
        nonlocal real_config, cached_real_mons, cached_real_conf

        if kwargs:
            real_config.update(kwargs)

        if cached_real_mons and real_config == cached_real_conf:
            return cached_real_mons

        # Region mapping
        rm = RegionMapping.from_file(real_config['rm_fname'])
        rm.connectivity = real_conn

        # Sensori
        sensorsEEG = SensorsEEG.from_file(real_config['sensors_fname'])
        sensorsEEG.configure()

        # Proiezione
        prEEG = ProjectionSurfaceEEG.from_file(
            real_config['projection_fname'],
            matlab_data_name=real_config['projection_matlab_name']
        )
        prEEG.sensors = sensorsEEG

        # Monitor
        mon_eeg = monitors.EEG(
            sensors=sensorsEEG,
            projection=prEEG,
            region_mapping=rm,
            period=real_config['period'],
            reference=None,
            variables_of_interest=np.array([0])
        )
        mon_raw = monitors.Raw()
        mon_progress = monitors.ProgressLogger(period=10000)

        mon_eeg.configure()
        mon_raw.configure()
        mon_progress.configure()

        cached_real_mons = (mon_eeg, mon_raw, mon_progress)
        cached_real_conf = real_config.copy()
        return cached_real_mons

    return create_real_eeg_monitor

def eeg_monitor_factory():
    """Factory che gestisce un monitor EEG con configurazione persistente."""
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

import zipfile
import os

#root_path = r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\sub-CLB00044_Connectome"

#with zipfile.ZipFile('my_connectivity.zip', 'w') as zf:
#    zf.write(os.path.join(root_path, 'centres.txt'))
#    zf.write(os.path.join(root_path, 'weights.txt'))
#    zf.write(os.path.join(root_path, 'tract_lengths.txt'))  # attenzione: TVB si aspetta "tract_lengths.txt"

#conn = get_conn(filename=r'C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Coding\my_connectivity.zip')

#print(conn.weights.shape)

# Carica il .MAT e ispeziona le chiavi
mat = scipy.io.loadmat(r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\sub-CLB00044_EEGProjection.mat")
print(mat.keys())  # vedi come si chiama la matrice dentro

print(mat['__header__'])   # info sull'header del file
print(mat['__version__'])  # versione del formato .MAT
print(mat['__globals__'])
print(mat['ProjectionMatrix'].shape)  # dimensioni della matrice

conn = get_conn(filename=r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\HC_data\real_patient_connectome.zip")
rm_f_name = (r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\sub-CLB00044_region_mapping.txt")
rm = RegionMapping.from_file(rm_f_name)
rm.connectivity=conn
sensorsEEG = SensorsEEG.from_file(r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\sub-CLB00044_EEG_Locations.txt")
sensorsEEG.configure()
prEEG = ProjectionSurfaceEEG.from_file(r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\sub-CLB00044_EEGProjection.mat", matlab_data_name="ProjectionMatrix")
prEEG.sensors=sensorsEEG
mons = ( monitors.EEG(sensors=sensorsEEG, projection=prEEG, region_mapping=rm, period=temp_avg_period, reference=None, variables_of_interest=np.array([0])),monitors.Raw(),monitors.ProgressLogger(period=10000),)
mons[0].configure()
mons[1].configure()
mons[2].configure()

from tvb.datatypes.surfaces import CorticalSurface
from tvb.datatypes.cortex import Cortex
from tvb.datatypes.region_mapping import RegionMapping
import numpy as np
import tvb.simulator.lab as tvb
from classes import JRPSP

CORTEX_DIR = r'C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\sub-CLB00044_Cortex'

# Surface
vertices  = np.loadtxt(os.path.join(CORTEX_DIR, 'vertices.txt'))
triangles = np.loadtxt(os.path.join(CORTEX_DIR, 'triangles.txt'), dtype=int)
normals   = np.loadtxt(os.path.join(CORTEX_DIR, 'normals.txt'))

print(vertices.shape)   # atteso (N, 3)
print(triangles.shape)  # atteso (M, 3)
print(normals.shape)    # atteso (N, 3)

import numpy as np

tetra_map = np.load(r"C:\Users\User\OneDrive - University of Pisa\Desktop\prova_disperata\tetra_to_roi_mapping.npy")
print(tetra_map[:10])   # prime 10 righe
print(tetra_map[-10:])  # ultime 10 righe

# Quante ROI uniche nella colonna 1?
print("ROI uniche:", np.unique(tetra_map[:, 1]))
print("Nr ROI:", len(np.unique(tetra_map[:, 1])))

# Quanti vertici unici nella colonna 0?
print("Vertici unici:", len(np.unique(tetra_map[:, 0])))

surface = CorticalSurface()
surface.vertices = vertices
surface.triangles = triangles
surface.vertex_normals = normals
surface.configure()
sim_length = 12000
model_pars = dict(
        A=3.25, B=22, v0=6.0,
        a=0.1, b=0.05, r=0.56,
        nu_max=0.0025, J=128,
        a_1=1.0, a_2 = 0.8, #a_2 around 0.84375
        a_3=0.25, a_4 = 0.25, #a_4 around 0.26367
        mu=0.22
    )

sim = Simulator_Modified(        
    model=JRPSP(
            variables_of_interest=("y0", "y1", "y2", "y3", "y4", "y5", "y6", "y7"),
            **{k: np.array(v) for k, v in model_pars.items()}
        ),        
    connectivity=conn,      
    coupling=tvb.coupling.Linear(a=np.r_[0.1 * 10]),     
    conduction_speed=float(conn.speed),       
    integrator=tvb.integrators.EulerStochastic(dt=0.1),     
    monitors=mons,        
    surface=None,        
 #   stimulus=input_stimulus,  
    simulation_length=sim_length
) 
sim.configure()
    #print(sim.model.__dict__)
results = sim.run(simulation_length=sim_length)
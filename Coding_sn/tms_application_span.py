import os

import numpy as np
import argparse
from tvb.simulator.lab import *
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils import skin_and_sensors, brain_activity, simulate, preprocess, eeg_plot, simulate_different_combination_parameters, simulate_parametri_lorenzo
from utils import functional_connectivity, psd, functional_connectivity_band, functional_connectivity_img_coh, psd_simil_lorenzo, Simulate2, functional_connectivity_lorenzo
from utils import make_train, simulate_with_stimulus, stimulus_visualization, new_parcellation_simulate, deterministic_new_parcellation_simulate
from classes import MultiStimuliRegion
from data_analysis_utils import real_vs_simulated_FC, real_vs_simulated_PSD, NSGAII, analyze_regular_data, analyze_reduced_data, load_simulation_results
import global_variable_creation as gv
from tms_application_utils import tms_seeds_analysis
import subprocess
import numpy as np
import joblib
from joblib import Parallel, delayed

from tqdm import tqdm

stages_couples = [("0","1"), ("0","3"), ("1","0"), ("3","0"), ("5","0"), ("5","3"), ("1","1"), ("3","3"), ("5","5")]
frequence = ["--high_f", "--low_f"]
application_spots = ["0", "1", "2", "3", "4"]

from contextlib import contextmanager

@contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()

def launch_simulation(f, i, j, k):
    print(f"\n>>> Simulating for tau = {i} abeta = {j} application spot {k} and frequence {f} ...")

    comando = [
        "python", 
        "tms_application.py", 
        f,
        "--abeta", i,
        "--tau", j,
        "--spot", k,
    ]

    # Rimuovendo capture_output, vedrai i messaggi di main.py in tempo reale
    risultato = subprocess.run(comando) 
    if risultato.returncode == 0:
        print(f">>> Simulazione (tau,abeta,application spot, frequence) = ({i},{j},{k},{f}) completata con successo.")
    else:
        print(f">>> Errore durante la simulazione (tau,abeta,application spot, frequence) = ({i},{j},{k},{f}).")

tasks = [(f, i, j, k) for f in frequence for (i,j) in stages_couples for k in application_spots]

for f, i, j, k in tqdm(tasks):
    launch_simulation(f, i, j, k)

"""
from tqdm import tqdm

with tqdm_joblib(tqdm(total=len(tasks))) as progress_bar:
    Parallel(n_jobs=2)(
        delayed(launch_simulation)(f, i, j, k) for f, i, j, k in tasks
    )
"""
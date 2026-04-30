from tvb.simulator.lab import *

import subprocess
import joblib
from joblib import Parallel, delayed

from tqdm import tqdm

from concurrent_log_handler import ConcurrentRotatingFileHandler
handler = ConcurrentRotatingFileHandler('logs/library.log', maxBytes=10**6, backupCount=5)

stages_couples = [("3", "3")]# [("0","1"), ("0","3"), ("1","0"), ("3","0"), ("5","0"), ("5","3"), ("1","1"), ("3","3"), ("5","5")]
frequence = ["--high"]#["--low_f", "--high_f"]#, "--low_f"]
application_spots = ["4"]#, "5", "6", "7"]#["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
seeds = ["0"]#, "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"]
intensities = ["8"]#["0.5", "2", "4", "6", "8"]#["1", "10", "100", "500"]#, "2", "4", "6"#["0.5", "2", "4", "6", "8"]#

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

def launch_simulation(f, i, j, k, l, a):
    print(f"\n>>> Simulating for abeta = {i} tau = {j} application spot {k} frequence {f} seed {l} and intensity {a}...")

    comando = [
        "python", 
        "tms_application.py",
        "--show_plots",
        "--save_folder", f"Prova_boh", #f"TMS_n02_th18_new_var5\Ab{i}t{j}s{k}f{f}i{a}", 
        f,
        "--tms",
        "--tms_intensity", a,
        #"--FC",
        "--PSD",
        "--abeta", i,
        "--tau", j,
        "--spot", k,
        "--rnd", l,
    ]

    risultato = subprocess.run(comando) 
    if risultato.returncode == 0:
        print(f">>> Simulazione (abeta, tau, application spot, frequence, seed, intensity) = ({i},{j},{k},{f},{l},{a}) completata con successo.")
    else:
        print(f">>> Errore durante la simulazione (abeta, tau, application spot, frequence, seed, intensity) = ({i},{j},{k},{f},{l},{a}).")

tasks = [(f, i, j, k, l, a) for f in frequence for (i,j) in stages_couples for k in application_spots for l in seeds for a in intensities]
"""
for f, i, j, k in tqdm(tasks):
    launch_simulation(f, i, j, k)

"""
from tqdm import tqdm

with tqdm_joblib(tqdm(total=len(tasks))) as progress_bar:
    Parallel(n_jobs=-4)(
        delayed(launch_simulation)(f, i, j, k, l, a) for f, i, j, k, l, a in tasks
    )

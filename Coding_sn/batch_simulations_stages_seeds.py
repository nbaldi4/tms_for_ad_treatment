import subprocess
import numpy as np
import joblib
from joblib import Parallel, delayed

stages = np.array(["0", "1", "2", "3", "4", "5"])
seeds = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39"])


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

def launch_simulation(i, j, k):
    print(f"\n>>> Simulating for tau = {i} abeta = {j} and random seed {k}...")

    comando = [
        "python", 
        "main.py", 
        "--PSD",
        "--FC",
        "--conn_sim",
        "--save_plots",
        "--save_folder", f'Stages_simnibs\Stage_{i}_{j}',
        "--tau", i,
        "--abeta", j,
        "--rnd", k,
    ]

    # Rimuovendo capture_output, vedrai i messaggi di main.py in tempo reale
    risultato = subprocess.run(comando) 
    if risultato.returncode == 0:
        print(f">>> Simulazione (tau,abeta,seed) = ({i},{j},{k}) completata con successo.")
    else:
        print(f">>> Errore durante la simulazione (tau,abeta,seed) = ({i},{j},{k}).")

tasks = [(i, j, k) for i in stages for j in stages for k in seeds]

from tqdm import tqdm

with tqdm_joblib(tqdm(total=len(tasks))) as progress_bar:
    Parallel(n_jobs=-3)(
        delayed(launch_simulation)(i, j, k) for i, j, k in tasks
    )
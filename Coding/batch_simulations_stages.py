import subprocess
import numpy as np
import joblib
from joblib import Parallel, delayed

stages = np.array(["0", "1", "2", "3", "4", "5"])


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

def launch_simulation(i, j):
    print(f"\n>>> Simulating for tau = {i} and abeta = {j}...")

    comando = [
        "python", 
        "main.py", 
        "--PSD",
        "--FC",
        "--save_plots",
        "--save_folder", 'Stages_16_02_N',
        "--tau", i,
        "--abeta", j,
        #"--save_raw_data",
    ]

    # Rimuovendo capture_output, vedrai i messaggi di main.py in tempo reale
    risultato = subprocess.run(comando) 
    if risultato.returncode == 0:
        print(f">>> Simulazione (tau,abeta) = ({i},{j}) completata con successo.")
    else:
        print(f">>> Errore durante la simulazione (tau,abeta) = ({i},{j}).")

tasks = [(i, j) for i in stages for j in stages]

from tqdm import tqdm

with tqdm_joblib(tqdm(total=len(tasks))) as progress_bar:
    Parallel(n_jobs=-3)(
        delayed(launch_simulation)(i, j) for i, j in tasks
    )
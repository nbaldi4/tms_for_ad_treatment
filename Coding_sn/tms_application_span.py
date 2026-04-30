from tvb.simulator.lab import *
import subprocess
import joblib

from tqdm import tqdm

stages_couples = [("0","1"), ("0","3"), ("1","0"), ("3","0"), ("5","0"), ("5","3"), ("1","1"), ("3","3"), ("5","5")] #(abeta, tau)
frequence = ["--high_f", "--low_f"]
application_spots = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

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
    print(f"\n>>> Simulating for abeta = {i} tau = {j} application spot {k} and frequence {f} ...")

    comando = [
        "python",
        "tms_application.py", 
        f,
        "--int_an",
        #"--test",
        "--abeta", i,
        "--tau", j,
        "--spot", k,
    ]

    # Rimuovendo capture_output, vedrai i messaggi di main.py in tempo reale
    risultato = subprocess.run(comando) 
    if risultato.returncode == 0:
        print(f">>> Simulazione (abeta, tau, application spot, frequence) = ({i},{j},{k},{f}) completata con successo.")
    else:
        print(f">>> Errore durante la simulazione (abeta, tau, application spot, frequence) = ({i},{j},{k},{f}).")

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
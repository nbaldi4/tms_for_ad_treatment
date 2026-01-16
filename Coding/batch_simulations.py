import subprocess
import numpy as np
from joblib import Parallel, delayed

stages = np.array(["0", "1", "2", "3", "4", "5"])

def launch_simulation(i, j):
    print(f"\n>>> Simulating for tau = {i} and abeta = {j}...")

    comando = [
        "python", 
        "main.py", 
        "--PSD",
        "--FC",
        "--save_plots",
        "--save_folder", 'Stages_16_01_2026',
        "--tau", i,
        "--abeta", j
    ]

    # Rimuovendo capture_output, vedrai i messaggi di main.py in tempo reale
    risultato = subprocess.run(comando) 
    if risultato.returncode == 0:
        print(f">>> Simulazione (tau,abeta) = ({i},{j}) completata con successo.")
    else:
        print(f">>> Errore durante la simulazione (tau,abeta) = ({i},{j}).")

tasks = [(i, j) for i in stages for j in stages]

Parallel(n_jobs=-3)(
        delayed(launch_simulation)(i, j) for i, j in tasks
    )
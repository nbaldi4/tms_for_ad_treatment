import subprocess
import numpy as np
import joblib
from joblib import Parallel, delayed
import time
from tqdm import tqdm

import os
from itertools import product

# 1. Configurazione percorsi
root_folder = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Prova_28_01"

# 2. Definiamo i range (se sono da 0 a 5, usa range(6); se sono da 0 a 6, usa range(7))
# Basandoci sul tuo messaggio, ipotizziamo range(6) per x, y, z, k, q
valori = range(6) 

# Generiamo tutte le combinazioni teoriche (xyzkq)
combinazioni_teoriche = list(product(valori, valori, valori, valori, valori))

# 3. Leggiamo le cartelle esistenti
cartelle_reali = os.listdir(root_folder)

# Creiamo un set delle combinazioni già completate estraendole dai nomi delle cartelle
completate = set()
for cartella in cartelle_reali:
    # Cerchiamo di estrarre i numeri dal nome tipo: a0_b_abeta0_tau0_J0_weights1
    try:
        # Questo split rompe la stringa nei punti chiave
        # a0 -> 0, abeta0 -> 0, tau0 -> 0, J0 -> 0, weights1 -> 1
        parti = cartella.split('_')
        x = int(parti[0].replace('a', ''))
        y = int(parti[2].replace('abeta', ''))
        z = int(parti[3].replace('tau', ''))
        k = int(parti[4].replace('J', ''))
        q = int(parti[5].replace('weights', ''))
        completate.add((x, y, z, k, q))
    except (IndexError, ValueError):
        # Ignora file o cartelle che non seguono il pattern
        continue

# 4. Calcolo delle mancanze
mancanti = [c for c in combinazioni_teoriche if c not in completate]

# 5. Output dei risultati
print(f"Combinazioni totali attese: {len(combinazioni_teoriche)}")
print(f"Combinazioni trovate: {len(completate)}")
print(f"Combinazioni mancanti: {len(mancanti)}")
print(len(mancanti))


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

tau_stages = np.array([0, 1, 2, 3, 4, 5])
abeta_stages = np.array([2, 3, 4, 5])

tau_combination = np.zeros((216, 3))
tau_index = 0
for i in tau_stages:
    for j in tau_stages:
        for k in tau_stages:
            tau_combination[tau_index] = [i, j, k] # b J w
            tau_index += 1
            
abeta_combination = np.zeros((216, 2))
abeta_index = 0
for i in abeta_stages:
    for j in abeta_stages:
        abeta_combination[abeta_index] = [i, j] #a, b
        abeta_index += 1

def launch_simulation(x, y, z, k, q):
    t0 = time.perf_counter()
    print(
    f">>> Simulating for a={x}, b_abeta={y}, b_tau={z}, J={k}, w={q} "
    )

    comando = [
        "python", 
        "main.py", 
        "--PSD",
        "--FC",
        "--save_plots",
        "--save_folder", 'Prova_28_01',
        "--tau_array", 
        str(int(z)), #z
        str(int(k)), #k
        str(int(q)), #q
        "--abeta_array",
        str(int(x)), #x
        str(int(y)) #y
    ]

    # Rimuovendo capture_output, vedrai i messaggi di main.py in tempo reale
    risultato = subprocess.run(comando) 

    t1 = time.perf_counter()
    elapsed = t1 - t0
    if risultato.returncode == 0:
        print(f">>> Simulazione (tau,abeta) = ( a={x}, b_abeta={y}, b_tau={z}, J={k}, w={q}) completata con successo.")
    else:
        print(f">>> Errore durante la simulazione (tau,abeta) = ( a={x}, b_abeta={y}, b_tau={z}, J={k}, w={q}).")

    return elapsed

tasks = [
    (x, y, z, k, q)
    for (x, y, z, k, q) in mancanti
]

start = time.perf_counter()

with tqdm_joblib(tqdm(total=len(tasks), desc="Simulazioni", unit="sim")):
    times = Parallel(n_jobs=-3)(
        delayed(launch_simulation)(x, y, z, k, q) for x, y, z, k, q in tasks
    )

end = time.perf_counter()

times = np.array(times)

print("\n========== STATISTICHE ==========")
print(f"Simulazioni totali : {len(times)}")
print(f"Tempo medio         : {times.mean():.2f} s")
print(f"Tempo minimo        : {times.min():.2f} s")
print(f"Tempo massimo       : {times.max():.2f} s")
print(f"Tempo totale reale  : {end - start:.2f} s")
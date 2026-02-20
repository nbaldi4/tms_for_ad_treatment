import itertools
import os
import time
from threading import Lock
from joblib import Parallel, delayed
from tqdm import tqdm
import joblib

DRY_RUN = True

# =========================
# PARAMETER SPACE
# =========================
tau_all = [0, 1, 2]
abeta_all = [0, 1]

all_simulations = set(
    (tau, abeta)
    for tau in itertools.product(tau_all, repeat=3)
    for abeta in itertools.product(abeta_all, repeat=2)
)

# =========================
# DONE FILE
# =========================
DONE_FILE = "done.txt"
lock = Lock()

def load_done():
    done = set()
    if os.path.exists(DONE_FILE):
        with open(DONE_FILE) as f:
            for line in f:
                tau, abeta = line.strip().split(";")
                done.add((
                    tuple(map(int, tau.split(","))),
                    tuple(map(int, abeta.split(",")))
                ))
    return done

def mark_done(tau, abeta):
    with lock:
        with open(DONE_FILE, "a") as f:
            f.write(f"{','.join(map(str,tau))};{','.join(map(str,abeta))}\n")

# =========================
# SIMULATION FUNCTION (GLOBAL)
# =========================
def launch_simulation(sim):
    tau, abeta = sim
    if DRY_RUN:
        print(f"WOULD RUN  tau={tau}  abeta={abeta}")
    else:
        print(f"RUNNING tau={tau}  abeta={abeta}")
    mark_done(tau, abeta)
    time.sleep(0.01)

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    done_simulations = load_done()
    remaining_simulations = sorted(all_simulations - done_simulations)

    print(f"Totali: {len(all_simulations)}  Fatto: {len(done_simulations)}  Da fare: {len(remaining_simulations)}")

    # TQDM + Joblib senza context manager complicati
    with tqdm(total=len(remaining_simulations)) as pbar:
        def launch_and_update(sim):
            launch_simulation(sim)
            pbar.update(1)

        Parallel(n_jobs=4)(delayed(launch_and_update)(sim) for sim in remaining_simulations)

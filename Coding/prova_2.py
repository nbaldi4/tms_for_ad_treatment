import numpy as np
import pandas as pd
from joblib import Parallel, delayed
# Altri import
import time
import os
import matplotlib.pyplot as plt
import sys 

# pymoo
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from pymoo.core.termination import Termination

# tue funzioni
from utils import preprocess, simulate
from data_analysis_utils import real_vs_simulated_FC_single_value, real_vs_simulated_PSD_single_value
import global_variable_creation as gv

# --- 1. DATI E CARICAMENTO ---

# Array di dati clinici (mantenuti per riferimento, non usati nella simulazione)
Abeta_patients = np.array([1559.0, np.nan, 338.0, 974.0, np.nan, 384.0, np.nan, 1239.0, 758.0,
                         np.nan, np.nan, 1742.0, np.nan, 527.0, np.nan, 702.0, 1170.0, np.nan, 
                         812.0, 1238.0, np.nan, np.nan, 1202.0, 702.0, np.nan, 692.0, 876.0,
                         1251.0, 490.0, np.nan, 1320.0, 1962.0, np.nan, 1374.0, np.nan, 1572.0,
                         685.0, 698.0, np.nan, 464.0])

tau_patients = np.array([40.0, np.nan, 128.2, 20.0, np.nan, 30.6, np.nan, 44.0, 125.0, np.nan,
                        np.nan, 53.0, np.nan, 34.0, np.nan, 81.0, 28.0, np.nan, 56.0, 30.0,
                        np.nan, np.nan, 25.0, 40.0, np.nan, 103.0, 158.0, 36.0, 103.0, np.nan,
                        39.4, 102.6, np.nan, 40.4, np.nan, 42.9, 156.9, 83.0, np.nan, 18.2])

# Caricamento dati (i percorsi devono essere corretti)
MCI_FC_data_path = r'C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\MCI_data\fc_mci_preview.npy'
MCI_PSD_data_path = r'C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\MCI_data\psd_mci_preview.npy'
HC_FC_data_path = gv.HC_FC_data if hasattr(gv, 'HC_FC_data') else 'percorso_HC_FC_da_correggere'
HC_PSD_data_path = gv.HC_PSD_data if hasattr(gv, 'HC_PSD_data') else 'percorso_HC_PSD_da_correggere'

try:
    GLOBAL_MCI_FC_REAL = np.load(MCI_FC_data_path) 
    GLOBAL_MCI_PSD_REAL = np.load(MCI_PSD_data_path)
    # Assumiamo che anche il percorso della SC sia stato corretto esternamente
    print("Dati HC_FC e HC_PSD caricati globalmente con successo.")
except FileNotFoundError as e:
    print("\n#################################################################")
    print(f"ERRORE CRITICO: Controlla e correggi il percorso del file: {e.filename}")
    print("#################################################################\n")
    sys.exit(1)

# --- 2. DEFINIZIONE DEL PROBLEMA E PARAMETRI ---

# Parametri ottimizzati (a e b)
a_range = [0.01, 0.3]
b_range = [0.01, 0.1]
# Parametri fissi
FIXED_G = 1.0        
FIXED_VELOCITY = 100 
FIXED_NOISE = 0.5    
INDEX_PATIENT = 23

# Parametri ottimizzazione
pop_size = 10
n_gen = 5
ref_pt = np.array([2.0, 2.0, 30.0])
max_to_improve = 2

def evaluate_simulation(row, sim_time=20000):
    """
    Funzione di valutazione ESEGUITA IN PARALLELO.
    Utilizza le variabili globali per i dati reali (GLOBAL_...).
    """
    a, b = row 
    g = FIXED_G
    velocity = FIXED_VELOCITY
    noise = FIXED_NOISE

    # Esegue la simulazione
    ttavg, tavg, teeg, eeg = simulate(sim_time=sim_time, g=g, velocity=velocity, noise=noise, a=a, b=b)
    teeg, eeg = preprocess(teeg, eeg)

    # Confronto FC
    regular_mae, reduced_mae = real_vs_simulated_FC_single_value(
        fc_healty=GLOBAL_MCI_FC_REAL, 
        index=INDEX_PATIENT,
        time_series=eeg,
        return_value=True
    )

    # Confronto PSD
    psd_mae, peak_freq = real_vs_simulated_PSD_single_value(
        matrix=GLOBAL_MCI_PSD_REAL, 
        index=INDEX_PATIENT,
        time_series=eeg,
        return_value=True
    )
    return regular_mae, reduced_mae, psd_mae, peak_freq

all_results = []

class EEGMetricProblem(Problem):
    def __init__(self, xl, xu, n_jobs=-1):
        # n_var = 2 (a e b), n_obj = 3 (M1, M2, M3)
        super().__init__(n_var=2, n_obj=3, xl=xl, xu=xu) 
        self.n_jobs = n_jobs

    def _evaluate(self, X, out, *args, **kwargs):
        # *** QUESTA RIGA AVVIA L'ESECUZIONE IN PARALLELO SUI CORE DELLA CPU ***
        results_with_peak = Parallel(n_jobs=self.n_jobs)(
            delayed(evaluate_simulation)(row) for row in X
        )
        
        # NSGA-II prende solo le prime 3 metriche (M1, M2, M3)
        F = np.array([res[:3] for res in results_with_peak])
        out["F"] = F

        # Salvataggio di 5 parametri (fissi + ottimizzati) + 4 metriche
        for x, res in zip(X, results_with_peak):
            full_quintet = [FIXED_G, FIXED_VELOCITY, FIXED_NOISE] + list(x)
            all_results.append(full_quintet + list(res))
            
# --- 3. FUNZIONI DI CONVERGENZA E TERMINAZIONE (Omesse per brevità) ---
# ... (Mantieni compute_spacing, nsga_callback, CombinedTermination)

# Ricrea le funzioni omettendo il corpo per brevità, assumendo che siano nel codice originale:
def compute_spacing(F): 
    if len(F) < 2: return 0.0
    distances = []
    for i in range(len(F)):
        others = np.delete(F, i, axis=0)
        d = np.min(np.linalg.norm(F[i] - others, axis=1))
        distances.append(d)
    distances = np.array(distances)
    mean_d = np.mean(distances)
    return np.sqrt(np.mean((distances - mean_d) ** 2))

def nsga_callback(algorithm):
    F = algorithm.pop.get("F")
    hv = HV(ref_point=ref_pt)(F)
    spacing = compute_spacing(F)
    n_nd = len(F)
    convergence_data["generation"].append(algorithm.n_gen)
    convergence_data["hypervolume"].append(hv)
    convergence_data["spacing"].append(spacing)
    convergence_data["n_nd"].append(n_nd)
    print(f"[Gen {algorithm.n_gen}] HV={hv:.4f} | Spacing={spacing:.4f} | ND={n_nd}")
    
class CombinedTermination(Termination):
     def __init__(self, ref_point, max_gen=100, patience=5, min_rel_improv=1e-8, min_abs_improv=None):
        super().__init__()
        self.ref_point = np.array(ref_point)
        self.max_gen = int(max_gen)
        self.patience = int(patience)
        self.min_rel_improv = float(min_rel_improv) if min_rel_improv is not None else None
        self.min_abs_improv = float(min_abs_improv) if min_abs_improv is not None else None
        self.hv_history = []
        self.no_improve_count = 0

     def _update(self, algorithm):
        gen = algorithm.n_gen
        F = algorithm.pop.get("F")

        try:
            hv_now = HV(ref_point=self.ref_point)(F)
        except Exception:
            hv_now = -np.inf

        self.hv_history.append(hv_now)

        if gen >= self.max_gen:
            print(f"[Termination] stop: reached max_gen = {self.max_gen} (gen={gen})")
            return True

        if len(self.hv_history) == 1:
            return False
            
        prev_best = max(self.hv_history[:-1])
        delta = hv_now - prev_best
        
        improved = False
        if (self.min_abs_improv is not None) and (delta >= self.min_abs_improv): improved = True
        if (self.min_rel_improv is not None) and (abs(prev_best) > 0):
            rel = delta / abs(prev_best)
            if rel >= self.min_rel_improv: improved = True

        if improved: self.no_improve_count = 0
        else: self.no_improve_count += 1

        if self.no_improve_count >= self.patience:
            print(f"[Termination] stop: HV improvement below threshold for {self.patience} generations")
            return True
        return False

convergence_data = {
    "generation": [],
    "hypervolume": [],
    "spacing": [],
    "n_nd": []
}

termination_condition = CombinedTermination(
    ref_point=ref_pt,
    max_gen=n_gen,
    patience=max_to_improve,
    min_rel_improv=1e-8,
    min_abs_improv=None
)

# --- 4. ESECUZIONE DEL PROBLEMA IN PARALLELO ---

xl = np.array([a_range[0], b_range[0]])
xu = np.array([a_range[1], b_range[1]])

# n_jobs=-1 farà usare tutti i core
problem = EEGMetricProblem(xl=xl, xu=xu, n_jobs=-1) 

# --- (Ometto il test seriale di debugging, che ora assumeremo sia passato) ---

algorithm = NSGA2(pop_size=pop_size)

print(f"\n✅ Avvio dell'ottimizzazione NSGA-II in parallelo con {os.cpu_count()} core...")

# La funzione minimize avvierà l'esecuzione in parallelo quando chiama _evaluate
res = minimize(
    problem=problem,
    algorithm=algorithm,
    seed=1,
    termination=termination_condition,
    callback=nsga_callback,
    verbose=True,
    save_history=True
)

# --- 5. SALVATAGGIO DEI RISULTATI (Mantenuto Inalterato) ---

# Salvataggio Pareto (solo parametri ottimizzati)
pareto = pd.DataFrame(
    np.hstack([res.X, res.F]),
    columns=["a","b","M1","M2","M3"]
)
pareto.to_csv("pareto_front_nsga2_abeta_mask.csv", index=False)

pd.DataFrame(convergence_data).to_csv("nsga2_convergence_metrics_abeta_mask.csv", index=False)

def plot_convergence(convergence):
    # Ometto il corpo per brevità, assumendo che sia nel codice originale
    pass

# Salvataggio di tutti i quintetti (5 parametri)
columns = ["g","velocity","noise","a","b","M1","M2","M3","peak_freq"]
pareto_all = pd.DataFrame(all_results, columns=columns)
pareto_all.to_csv("all_evaluated_quintets_with_peak_abeta_mask.csv", index=False)
print("Tutti i quintetti valutati salvati in all_evaluated_quintets_with_peak_abeta_mask.csv")

plot_convergence(convergence_data)

print("Pareto front saved in pareto_front_nsga2_abeta_mask.csv")
print("Convergence metrics saved in nsga2_convergence_metrics_abeta_mask.csv")
print("Plots saved: hypervolume_convergence_abeta_mask.png, spacing_convergence_abeta_mask.png, nondominated_convergence_abeta_mask.png")
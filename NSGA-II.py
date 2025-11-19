# Ottimized workflow: LHS (parallel) -> Train surrogates -> NSGA-II using surrogates -> Validate Pareto with real simulations
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
import shap
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
import multiprocessing
import time
import os

# import delle tue funzioni (assumo che siano disponibili)
from utils import NSGA_simulate, plot_real_vs_simulated_psd_bar
from data_analysis_utils import NSGA_simulation_vs_real_data_FC
import global_variable_creation as gv

# -----------------------
# 1) Parametri utente
# -----------------------
# se vuoi usare tau_e e tau_i al posto di a,b -> basta cambiare nomi e ranges
g_range = [0.1, 5]
velocity_range = [1, 20]
noise_range = [0.001, 2]
a_range = [0.05, 0.150]   # -> potrebbe essere tau_e
b_range = [0.025, 0.075]  # -> potrebbe essere tau_i

# dimensione LHS (ridotta per velocità): se vuoi massima esplorazione aumenta
N_LHS = 10 #500

# NSGA settings
POP_SIZE = 10 #50
N_GEN = 10 #20

# quanti soluzioni Pareto vogliamo validare con simulazione reale (cap)
MAX_VALIDATE = 10 #100

# num jobs per joblib (-1 = tutti i core)
N_JOBS = -1

# -----------------------
# 2) Funzione LHS (log-scale optional)
# -----------------------
def lhs_sample(N, ranges, rng=np.random.default_rng(42)):
    dims = len(ranges)
    u = np.zeros((N, dims))
    for j, (low, high) in enumerate(ranges):
        # stratified samples
        points = (np.arange(N) + rng.random(N)) / N #mi viene il dubbio che possano essere numeri da 1 a 2N
        rng.shuffle(points)
        # se molti ordini di grandezza, campiono su log scale
        if low > 0 and (high / low > 10):
            points = 10**(np.log10(low) + points * (np.log10(high)-np.log10(low)))
        else:
            points = low + points * (high-low)
        u[:, j] = points
    cols = ["g","velocity","noise","a","b"]
    return pd.DataFrame(u, columns=cols)

# -----------------------
# 3) Funzione che esegue la simulazione (usata SOLO in LHS e validazione)
# -----------------------
def evaluate_simulation(row, sim_time=20000, return_full=False):
    """
    row: array-like [g, velocity, noise, a, b]
    return: tuple (M1, M2, M3) - le metriche obiettivo
    return_full True -> ritorna anche eeg/time-series se vuoi
    """
    g, velocity, noise, a, b = row
    # ESEGUI la simulazione pesante
    ttavg, tavg, teeg, eeg = NSGA_simulate(sim_time=sim_time, g=g, velocity=velocity, noise=noise, a=a, b=b)
    # confronta simulazione vs dati reali (puoi adattare i nomi/argomenti)
    regular_mae, reduced_mae = NSGA_simulation_vs_real_data_FC(fc_healty=np.load(gv.real_FC_data), sim_time=sim_time,
                                                               g=g, velocity=velocity, noise=noise, a=a, b=b)
    # calcolo PSD MAE (se plot_real_vs_simulated_psd_bar ritorna una metrica)
    psd_mae = plot_real_vs_simulated_psd_bar(path=r"C:\Users\User\OneDrive - University of Pisa\Desktop\TVB_tutorials\Dati_Healthy\psd_ctr_preview.npy",
                                            time_series=eeg)
    
    if return_full:
        return regular_mae, reduced_mae, psd_mae, (ttavg, tavg, teeg, eeg)
    else:
        return regular_mae, reduced_mae, psd_mae

# -----------------------
# 4) Esegui LHS e parallellizza le simulazioni
# -----------------------
lhs_df = lhs_sample(N_LHS, [g_range, velocity_range, noise_range, a_range, b_range])
print(f"LHS con {N_LHS} punti, simulazioni parallele su {N_JOBS} jobs")

start_lhs = time.time()
# prepare list of param rows
rows = lhs_df.values

# joblib parallel map
results = Parallel(n_jobs=N_JOBS, verbose=10)(
    delayed(evaluate_simulation)(row) for row in rows #delayed serve perchè altrimenti chiamando direttamente evaluate simulation
    # la funzione partirebbe npn appena chiamata e prima che possa essere distribuita fra i processori
)
results = np.array(results)
lhs_df["M1"] = results[:,0]
lhs_df["M2"] = results[:,1]
lhs_df["M3"] = results[:,2]
end_lhs = time.time()
print(f"LHS simulazioni finite in {end_lhs - start_lhs:.1f} s")

# salva LHS per riproducibilita
lhs_df.to_csv("lhs_results.csv", index=False)

# -----------------------
# 5) Costruisci surrogates (Random Forest)
# -----------------------
X = lhs_df[["g","velocity","noise","a","b"]].values
Y = lhs_df[["M1","M2","M3"]].values

rf_M1 = RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=42) # n_jobs=1 per evitare conflitti in parallelo negli step successivi
rf_M1.fit(X, Y[:,0])
rf_M2 = RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=42)
rf_M2.fit(X, Y[:,1])
rf_M3 = RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=42)
rf_M3.fit(X, Y[:,2])

print("Surrogate RF addestrati.")

# --- VALIDAZIONE DELLA RF SUL TRAINING SET ---
from sklearn.metrics import r2_score, mean_squared_error

for i, (rf, name) in enumerate(zip([rf_M1, rf_M2, rf_M3], ["M1","M2","M3"])):
    pred = rf.predict(X)
    r2 = r2_score(Y[:,i], pred)
    rmse = np.sqrt(mean_squared_error(Y[:,i], pred))
    print(f"RF {name}: R^2 = {r2:.3f}, RMSE = {rmse:.3f}")


# -----------------------
# 6) (Opzionale) SHAP per interpretabilità
# -----------------------
try:
    explainer_M1 = shap.TreeExplainer(rf_M1)
    shap_values_M1 = explainer_M1.shap_values(X)
    shap.summary_plot(shap_values_M1, X, feature_names=["g","velocity","noise","a","b"], show=True)
    # ripeti per M2,M3 se vuoi
except Exception as e:
    print("SHAP plot fallito (dipendenze/ambiente). Eccezione:", e)

# -----------------------
# 7) Definizione Problema per NSGA-II che usa i surrogate (NON chiamerà NSGA_simulate)
# -----------------------
class CorticalSurrogateProblem(Problem):
    def __init__(self, rf1, rf2, rf3, xl, xu):
        # n_var = 5, n_obj = 3
        super().__init__(n_var=5, n_obj=3, n_constr=0, xl=xl, xu=xu)
        self.rf1 = rf1
        self.rf2 = rf2
        self.rf3 = rf3

    def _evaluate(self, X, out, *args, **kwargs):
        """
        X: array (pop_size, n_var)
        Utilizziamo i RF per predire i tre obiettivi in batch (molto veloce).
        """
        # predizioni batch
        pred1 = self.rf1.predict(X)
        pred2 = self.rf2.predict(X)
        pred3 = self.rf3.predict(X)
        out["F"] = np.column_stack([pred1, pred2, pred3])

# limiti (usiamo i limiti ridotti se vuoi)
xl = np.array([g_range[0], velocity_range[0], noise_range[0], a_range[0], b_range[0]])
xu = np.array([g_range[1], velocity_range[1], noise_range[1], a_range[1], b_range[1]])

problem = CorticalSurrogateProblem(rf_M1, rf_M2, rf_M3, xl=xl, xu=xu)

# -----------------------
# 8) Esegui NSGA-II (sulle surrogate -> velocissimo)
# -----------------------
algorithm = NSGA2(pop_size=POP_SIZE)
print("Avvio NSGA-II sulle surrogate (questo è veloce)...")
start_nsga = time.time()
res = minimize(problem,
               algorithm,
               ('n_gen', N_GEN),
               verbose=True,
               save_history=True,
               seed=1)
end_nsga = time.time()
print(f"NSGA-II (surrogate) completato in {end_nsga - start_nsga:.1f} s")

# res.X => soluzioni Pareto (parametri)
# res.F => obiettivi predetti dai surrogate
pareto_params = res.X
pareto_preds = res.F
hist = res.history

# salva risultati surrogate
pd.DataFrame(np.hstack([pareto_params, pareto_preds]),
             columns=["g","velocity","noise","a","b","M1_pred","M2_pred","M3_pred"]).to_csv("pareto_surrogate.csv", index=False)

# -----------------------
# 9) Validazione finale delle soluzioni Pareto con simulazioni reali (parallelo)
# -----------------------
n_to_validate = min(len(pareto_params), MAX_VALIDATE)
print(f"Validazione finale di {n_to_validate} soluzioni Pareto usando simulazioni reali (parallelo).")
to_validate = pareto_params[:n_to_validate]

start_val = time.time()
val_results = Parallel(n_jobs=N_JOBS, verbose=10)(
    delayed(evaluate_simulation)(row) for row in to_validate
)
val_results = np.array(val_results)
end_val = time.time()
print(f"Validazione finale completata in {end_val - start_val:.1f} s")

# salva confronto predizioni vs simulazioni reali
df_cmp = pd.DataFrame(np.hstack([to_validate,
                                 pareto_preds[:n_to_validate],
                                 val_results]),
                      columns=["g","velocity","noise","a","b",
                               "M1_pred","M2_pred","M3_pred",
                               "M1_sim","M2_sim","M3_sim"])
df_cmp.to_csv("pareto_validation_comparison.csv", index=False)
print("Risultati salvati in pareto_validation_comparison.csv")

# -----------------------
# 10) Stampa sommaria e stima chiamate simulate risparmiate
# -----------------------
lhs_calls = N_LHS
nsga_sim_calls = 0  # perché NSGA ha usato surrogate
validation_calls = n_to_validate
total_sim_calls = lhs_calls + validation_calls

print("Stima chiamate a NSGA_simulate:")
print(f" - LHS: {lhs_calls}")
print(f" - NSGA-II (reale durante ottimizzazione): {nsga_sim_calls}")
print(f" - Validazione Pareto finale: {validation_calls}")
print(f" -> Totale simulazioni eseguite: {total_sim_calls}")

# suggerimento: fai una validazione extra finale su 5-10 soluzioni selezionate (diverse generazioni)

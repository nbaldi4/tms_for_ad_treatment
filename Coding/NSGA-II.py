import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import r2_score, mean_squared_error
import time
import os
import matplotlib.pyplot as plt

# pymoo
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from pymoo.core.termination import Termination

# tue funzioni
from utils import preprocess, simulate
from data_analysis_utils import real_vs_simulated_FC, real_vs_simulated_PSD
import global_variable_creation as gv


g_range = [0.1, 10] #[0.1, 1.1]
velocity_range = [1, 300]
noise_range = [0.1, 5] #[0.1, 5]
a_range = [0.05, 0.150]
b_range = [0.025, 0.075]

pop_size = 100
n_gen = 200
ref_pt = np.array([2.0, 2.0, 30.0])
max_to_improve = 2

def evaluate_simulation(row, sim_time=20000):
    g, velocity, noise, a, b = row

    ttavg, tavg, teeg, eeg = simulate(sim_time=sim_time, g=g, velocity=velocity, noise=noise, a=a, b=b)
    teeg, eeg = preprocess(teeg, eeg)

    regular_mae, reduced_mae = real_vs_simulated_FC(
        fc_healty=np.load(gv.HC_FC_data),
        time_series=eeg,
        return_value=True
    )

    psd_mae, peak_freq = real_vs_simulated_PSD(
        matrix=np.load(gv.HC_PSD_data),
        time_series=eeg,
        return_value=True
    )

    return regular_mae, reduced_mae, psd_mae, peak_freq

all_results = []

class EEGMetricProblem(Problem):
    def __init__(self, xl, xu, n_jobs=-1):
        super().__init__(n_var=5, n_obj=3, xl=xl, xu=xu)
        self.n_jobs = n_jobs

    def _evaluate(self, X, out, *args, **kwargs):
        results_with_peak = Parallel(n_jobs=self.n_jobs)(
            delayed(evaluate_simulation)(row) for row in X
        )
        
        # Fourth output is peakfreq and we do not use it
        F = np.array([res[:3] for res in results_with_peak])
        out["F"] = F

        # Save parameters quintet + metrics + peak freq
        for x, res in zip(X, results_with_peak):
            all_results.append(list(x) + list(res))


convergence_data = {
    "generation": [],
    "hypervolume": [],
    "spacing": [],
    "n_nd": []
}

def compute_spacing(F):
    """
    Compute minimum distance between pareto front points.
    """
    if len(F) < 2:
        return 0.0

    distances = []
    for i in range(len(F)):
        # minimum distance between i and all other points
        others = np.delete(F, i, axis=0)
        d = np.min(np.linalg.norm(F[i] - others, axis=1))
        distances.append(d)

    distances = np.array(distances)
    mean_d = np.mean(distances)

    return np.sqrt(np.mean((distances - mean_d) ** 2))


def nsga_callback(algorithm):
    F = algorithm.pop.get("F")

    # a ref_point is needed to define the worst possible value in metrics space
    hv = HV(ref_point=ref_pt)(F)
    spacing = compute_spacing(F)
    n_nd = len(F)

    convergence_data["generation"].append(algorithm.n_gen)
    convergence_data["hypervolume"].append(hv)
    convergence_data["spacing"].append(spacing)
    convergence_data["n_nd"].append(n_nd)

    print(f"[Gen {algorithm.n_gen}] HV={hv:.4f} | Spacing={spacing:.4f} | ND={n_nd}")
    
class CombinedTermination(Termination):
    """
    Termination based on:
      - either max_gen (stop if generation >= max_gen)
      - or hypervolume do not improve more than min_rel_improv (or min_abs_improv)
        for 'patience' consecutive generations.
    """
    def __init__(self,
                 ref_point,
                 max_gen=100,
                 patience=5,
                 min_rel_improv=1e-5,
                 min_abs_improv=None):
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
            # if HV can't be computed (ex. empty F), consider hv_now = -inf
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

        if (self.min_abs_improv is not None) and (delta >= self.min_abs_improv):
            improved = True

        if (self.min_rel_improv is not None) and (abs(prev_best) > 0):
            rel = delta / abs(prev_best)
            if rel >= self.min_rel_improv:
                improved = True

        # update non-improving counter
        if improved:
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        if self.no_improve_count >= self.patience:
            print(f"[Termination] stop: HV improvement below threshold for {self.patience} generations "
                  f"(last delta={delta:.3e}, min_rel={self.min_rel_improv}, min_abs={self.min_abs_improv})")
            return True

        return False

termination_condition = CombinedTermination(
    ref_point=ref_pt,
    max_gen=n_gen,
    patience=max_to_improve,
    # you choose relative or absolute value
    min_rel_improv=1e-8,
    min_abs_improv=None
)


xl = np.array([g_range[0], velocity_range[0], noise_range[0], a_range[0], b_range[0]])
xu = np.array([g_range[1], velocity_range[1], noise_range[1], a_range[1], b_range[1]])

problem = EEGMetricProblem(xl=xl, xu=xu, n_jobs=-1)

algorithm = NSGA2(pop_size=pop_size)

res = minimize(
    problem=problem,
    algorithm=algorithm,
    seed=1,
    termination=termination_condition,
    callback=nsga_callback,
    verbose=True,
    save_history=True
)

pareto = pd.DataFrame(
    np.hstack([res.X, res.F]),
    columns=["g","velocity","noise","a","b","M1","M2","M3"]
)
pareto.to_csv("pareto_front_nsga2_03_02.csv", index=False)

pd.DataFrame(convergence_data).to_csv("nsga2_convergence_metrics_03_02.csv", index=False)

def plot_convergence(convergence):
    gen = convergence["generation"]

    # ---- Hypervolume ----
    plt.figure(figsize=(8,5))
    plt.plot(gen, convergence["hypervolume"], marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    plt.title("Hypervolume convergence")
    plt.grid(True)
    plt.savefig("hypervolume_convergence_03_02.png", dpi=150)
    plt.close()

    # ---- Spacing ----
    plt.figure(figsize=(8,5))
    plt.plot(gen, convergence["spacing"], marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Spacing (Diversity)")
    plt.title("Diversity convergence (Spacing)")
    plt.grid(True)
    plt.savefig("spacing_convergence_03_02.png", dpi=150)
    plt.close()

    # ---- Numero ND ----
    plt.figure(figsize=(8,5))
    plt.plot(gen, convergence["n_nd"], marker="o")
    plt.xlabel("Generation")
    plt.ylabel("# non dominated solutions")
    plt.title("Non dominated solutions")
    plt.grid(True)
    plt.savefig("nondominated_convergence_03_02.png", dpi=150)
    plt.close()

columns = ["g","velocity","noise","a","b","M1","M2","M3","peak_freq"]
pareto_all = pd.DataFrame(all_results, columns=columns)
pareto_all.to_csv("all_evaluated_quintets_with_peak_03_02.csv", index=False)
print("Tutti i quintetti valutati salvati in all_evaluated_quintets_with_peak_03_02.csv")

plot_convergence(convergence_data)

print("Pareto front saved in pareto_front_nsga2_03_02.csv")
print("Convergence metrics saved in nsga2_convergence_metrics_03_02.csv")
print("Plots saved: hypervolume_convergence_03_02.png, spacing_convergence_03_02.png, nondominated_convergence_03_02.png")
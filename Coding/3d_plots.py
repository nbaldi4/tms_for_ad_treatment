import numpy as np
import matplotlib.pyplot as plt
import global_variable_creation as gv
from mpl_toolkits.mplot3d import Axes3D
from global_variable_creation import get_conn


pareto_front = np.loadtxt(r"..\Best_results_so_far\MAE\pareto_front_nsga2.csv", delimiter=",", skiprows=1)
metrics = pareto_front[:, -3:]
mean_metrics1 = np.mean(metrics[:,0])
mean_metrics2 = np.mean(metrics[:,1])
mean_metrics3 = np.mean(metrics[:,2])
metrics_deviations = np.ones(metrics.shape)
metrics_deviations[:,0] = (metrics[:,0] - np.ones(len(metrics[:,0]))*mean_metrics1)/mean_metrics1
metrics_deviations[:,1] = (metrics[:,1] - np.ones(len(metrics[:,1]))*mean_metrics2)/mean_metrics2
metrics_deviations[:,2] = (metrics[:,2] - np.ones(len(metrics[:,2]))*mean_metrics3)/mean_metrics3


# Figura
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter 3D
ax.scatter(metrics[:,0], metrics[:,1], metrics[:,2], c='royalblue', s=40)

# Etichette (opzionali)
ax.set_xlabel('Regular FC')
ax.set_ylabel('Reduced FC')
ax.set_zlabel('PSD')
plt.title("Pareto front")

plt.show()

rank_matrics = metrics_deviations[:,0] + metrics_deviations[:,1] + 2*metrics_deviations[:,2]
idx = np.argmin(rank_matrics)
print('best solution is: ', idx, '; with parameters values: ', np.array2string(pareto_front[idx, :5], precision=20, suppress_small=False))
print('and metrics values: ', metrics[idx, -3:])

all_evaluations = np.loadtxt(r"..\Best_results_so_far\MAE\all_evaluated_quintets_with_peak.csv", delimiter=",", skiprows=1)
g = all_evaluations[:, 0]
a = all_evaluations[:, 3]
b = all_evaluations[:, 4]
peak = all_evaluations[:, -1]

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    g, a, b,
    c=peak,               # colore basato sulla 4ª colonna
    cmap='viridis',    # mappa colori a scelta
    s=60               # dimensione punti
)

cb = fig.colorbar(scatter, ax=ax)
cb.set_label("Peak frequence distribution")

ax.set_xlabel("g")
ax.set_ylabel("a")
ax.set_zlabel("b")

plt.title("Grafico 3D con 4ª variabile come colore")
plt.show()

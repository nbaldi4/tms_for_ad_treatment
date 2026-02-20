import numpy as np
import matplotlib.pyplot as plt
import global_variable_creation as gv
from mpl_toolkits.mplot3d import Axes3D
from global_variable_creation import get_conn

def visualize_pareto_front(file_path = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Coding\pareto_front_nsga2_03_02.csv", parameters_number=6, metrics_limits = -3):

    pareto_front = np.loadtxt(file_path, delimiter=",", skiprows=1)
    if metrics_limits < -3:
        metrics_limits_dx = metrics_limits + 3
        metrics = pareto_front[:, metrics_limits:metrics_limits_dx]
    else:
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
    rank_matrics_2 = metrics_deviations[:,2]
    idx = np.argmin(rank_matrics)
    idx_2 = np.argmin(rank_matrics_2)
    print('best solution is: ', idx, '; with parameters values: ', np.array2string(pareto_front[idx, :parameters_number], precision=20, suppress_small=False))
    print('and metrics values: ', metrics[idx, -3:])
    print('best psd solution is: ', idx_2, '; with parameters values: ', np.array2string(pareto_front[idx_2, :parameters_number], precision=20, suppress_small=False))

def peak_plot(file_path = r"..\Best_results_so_far\MAE\all_evaluated_quintets_with_peak.csv"):

    all_evaluations = np.loadtxt(file_path, delimiter=",", skiprows=1)
    g = all_evaluations[:, 0]
    a = all_evaluations[:, 3]
    b = all_evaluations[:, 4]
    peak = all_evaluations[:, -1]

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        g, a, b,
        c=peak,            # colore basato sulla 4ª colonna
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

visualize_pareto_front()#file_path = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Coding\grid_search_2D_results.csv", parameters_number=2, metrics_limits=-4)
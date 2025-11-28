import numpy as np
import matplotlib.pyplot as plt
import global_variable_creation as gv
from global_variable_creation import get_conn


pareto_front = np.loadtxt(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\pareto_front_nsga2.csv", delimiter=",", skiprows=1)
metrics = pareto_front[:, -3:]
print(metrics)
mean_metrics1 = np.mean(metrics[:,0])
mean_metrics2 = np.mean(metrics[:,1])
mean_metrics3 = np.mean(metrics[:,2])
print(mean_metrics1)
print(mean_metrics2)
print(mean_metrics3)
metrics_deviations = metrics
metrics_deviations[:,0] = (metrics[:,0] - np.ones(len(metrics[:,0]))*mean_metrics1)/mean_metrics1
metrics_deviations[:,1] = (metrics[:,1] - np.ones(len(metrics[:,1]))*mean_metrics2)/mean_metrics2
metrics_deviations[:,2] = (metrics[:,2] - np.ones(len(metrics[:,2]))*mean_metrics3)/mean_metrics3
print(metrics_deviations)

rank_matrics = metrics_deviations[:,0] + metrics_deviations[:,1] + metrics_deviations[:,2]
print(rank_matrics)
idx = np.argmin(rank_matrics)
print(idx)
print(np.min(rank_matrics))
print(pareto_front[idx, :])
"""
matrix = np.load(gv.struct_path)[:, :, :, :]
conn = get_conn()
mm = matrix[0, 3, :, :]
mm2 = matrix[0, 4, :, :]
cp=15

for j in range (len(matrix[0,3,:,0])):
    for k in range (len(matrix[0,3,0,:])):
        plt.plot(matrix[15,:,j,k]/matrix[15,0,j,k])

plt.xlabel('np values')
plt.ylabel('Cij/Cij(cp=0)')
plt.title('Relative connectivity weights variations')
plt.show()


print("CONNECTIVITY TVB", conn.weights)
print("CONNECTIVITY LORE", mm)

print("DIFFERENCE", mm2 - mm)
"""
#np.savetxt(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\conn_lore.txt", mm, fmt="%.3f")
#np.savetxt(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\conn_tvb.txt", conn.weights, fmt="%.3f")
#np.savetxt(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\conn_diff.txt", conn.weights - mm, fmt="%.3f")

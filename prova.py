import numpy as np
import matplotlib.pyplot as plt
import global_variable_creation as gv
from global_variable_creation import get_conn

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

np.savetxt(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\conn_lore.txt", mm, fmt="%.3f")
np.savetxt(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\conn_tvb.txt", conn.weights, fmt="%.3f")
np.savetxt(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\conn_diff.txt", conn.weights - mm, fmt="%.3f")

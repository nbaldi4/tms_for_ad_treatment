import numpy as np
import argparse
import matplotlib.pyplot as plt
from data_analysis_utils import analyze_regular_data, analyze_reduced_data, simulation_vs_real_data_FC
import global_variable_creation as gv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_data", action="store_true", help="Show real data FC (regular and reduced size) on screen")
    parser.add_argument("--save_plots", action="store_true", help="Save plots to disk")
    parser.add_argument("--fig_folder", type=str, help="Path to save figures")
    parser.add_argument("--struct_conn", type=str, help="Path to structural connectivity file")
    parser.add_argument("--sim_vs_real", action="store_true", help="Perform comparison between simulation and real data")
    parser.add_argument("--th", type=float, help="Threshold for FC visualization")
    parser.add_argument("--temp_avg_period", type=float, help="Average integration step")
    parser.add_argument(
        "--impaired_regions",
        nargs="+",               # accepts one or more values
        type=int,                # convert them to integers
        help="List of impaired region indices"
    )

    args = parser.parse_args()
    gv.real_data = args.real_data
    gv.save_plots = args.save_plots
    gv.sim_vs_real = args.sim_vs_real
    gv.th = args.th
    gv.temp_avg_period = args.temp_avg_period

    if args.fig_folder:
        gv.fig_folder = args.fig_folder

    if args.struct_conn:
        gv.struct_path = args.struct_conn

    if args.impaired_regions:
        gv.impaired_regions = args.impaired_regions

    if gv.real_data == True:
        analyze_regular_data(np.load(gv.real_FC_data))
        analyze_reduced_data(np.load(gv.real_FC_data))

    if gv.sim_vs_real == True:
        simulation_vs_real_data_FC(np.load(gv.real_FC_data))

if __name__ == "__main__":
    main()


"""

mask = np.triu(np.ones(corr_mean.shape), k=1).astype(bool)

for i in range(len(fc_healty[:,0,0])):
    matrix = fc_healty[i, :, :]
    r, _ = pearsonr(corr_mean[mask], matrix[mask])
    print("Pattern correlation:", r)

sub_corr_mean = 

for i in range(len(fc_healty[:,0,0])):
    macro_matrix = macro_fc(matrix = fc_healty[i, :, :])
    r, _ = pearsonr(corr_mean[mask], matrix[mask])
    print("Pattern correlation:", r)

macro_fc_sim = macro_fc(matrix = corr_eeg)
macro_fc_real = macro_fc(matrix = corr_mean)
tvb_matrix_sub, real_matrix_sub = to_common_indices(corr_eeg, corr_mean)
compute_matrices_diff(tvb_matrix_sub, real_matrix_sub)
compute_matrices_diff(macro_fc_sim, macro_fc_real)

# Posso calcolare anche la matrice delle differenze elemento per elemento
mask = np.triu(np.ones(normal_corr_mean.shape), k=1).astype(bool)

for i in range(len(fc_healty[:,0,0])):
    matrix = fc_healty[i, :, :]
    r, _ = pearsonr(normal_corr_mean[mask], matrix[mask])
    print("Pattern correlation:", r)

diff = fc_healty[0, :, :] - normal_corr_mean
plt.imshow(diff, cmap='bwr', vmin=-1, vmax=1)
plt.title("Element-wise difference (new - mean)")
plt.colorbar()
plt.show()
matrix = corr_ee
mae = np.mean(np.abs(diff))
print("Mean absolute difference:", mae)
eps = 1e-6
safe_std = np.where(z_std < eps, np.nan, z_std)
z_diff = (matrix - corr_mean) / safe_std
n_outliers = np.sum(np.abs(z_diff) > 2)
perc_outliers = n_outliers/(len(fc_healty[0,0,:])*len(fc_healty[0,:,0]))
print(f"{n_outliers} edges differ by more than 2 SDs from the group mean")
print(f"{perc_outliers} percentage of outliers differ by more than 2 SDs from the group mean")
"""
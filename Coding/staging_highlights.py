import pandas as pd
from pathlib import Path
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import global_variable_creation as gv
import sns

from data_analysis_utils import real_vs_stages_PSD, real_vs_stages_FC, to_common_indices, macro_fc
from additional_utils import extract_common_names
from sklearn.feature_selection import mutual_info_regression
from specparam import SpectralGroupModel # Usiamo Group per matrici [n_regions, n_freqs]


# Rimuove il limite di larghezza della colonna (None = nessun limite)
pd.set_option('display.max_colwidth', None)
root_folder_global = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Stages_25_01_(new_staging)"
root_folder_global_save = r"C:\Users\User\OneDrive - University of Pisa\Desktop\Staging_del_25_01_(new_staging)"

def compute_mi_rank(X, y, target_name):
    # Calcolo della Mutual Information
    mi_scores = mutual_info_regression(X, y, discrete_features=False, random_state=42)
    
    # Creazione classifica
    mi_results = pd.Series(mi_scores, name="MI_Score", index=X.columns)
    mi_results = mi_results.sort_values(ascending=False)
    
    print(f"\n--- Classifica Metriche per {target_name} ---")
    print(mi_results)
    return mi_results

def aperiodic_slope_manual(psd_all, freqs, fmin=4, fmax=40):

    psd_all = np.asarray(psd_all, float)
    freqs = np.asarray(freqs, float)

    mask = (freqs >= fmin) & (freqs <= fmax)

    psd_all = psd_all[:, mask]
    freqs = freqs[mask]

    psd_all = np.maximum(psd_all, 1e-12)

    log_f = np.log10(freqs)

    slopes = []
    for psd in psd_all:
        slope, _ = np.polyfit(log_f, np.log10(psd), 1)
        slopes.append(slope)

    return np.nanmean(slopes)

def collect_files(root_folder_path, files_name):
    """
    Raccoglie tutti i file che iniziano con 'FC_EEG alpha band' in tutte le sottocartelle
    della cartella root, estraendo tau e abeta dal nome del file.
    """
    root_folder = Path(root_folder_path)
    data = []

    # regex flessibile per tau e abeta
    pattern = re.compile(r"tau[_]?(\d+)[_]?abeta[_]?(\d+)", re.IGNORECASE)

    # glob ricorsivo dalla root (trova tutto)
    for file in root_folder.rglob(files_name):
        match = pattern.search(file.name)
        if match:
            tau_val = int(match.group(1))
            abeta_val = int(match.group(2))
        else:
            tau_val = None
            abeta_val = None

        data.append({
            "tau": tau_val,
            "abeta": abeta_val,
            "folder_name": file.parent.name,  # cartella immediata
            "file_name": file.name,
            "file_path": str(file.resolve())
        })

    df = pd.DataFrame(data)
    return df

def file_extraction(files_group, tau_stage, abeta_stage):
    file_sel = files_group[(files_group["tau"] == tau_stage) & (files_group["abeta"] == abeta_stage)]
    file_path_sel = file_sel["file_path"]
    file_path_string = file_path_sel.iloc[0] 
    file_data = np.load(file_path_string)
    return file_data

def stage_metrics(root_folder = root_folder_global):
    alpha_band_files = collect_files(root_folder, files_name="FC_EEG alpha (8-13)Hz_*")
    beta_band_files = collect_files(root_folder, files_name="FC_EEG beta (13-30)Hz_*")
    delta_band_files = collect_files(root_folder, files_name="FC_EEG delta (0.5-4)Hz_*")
    theta_band_files = collect_files(root_folder, files_name="FC_EEG theta (4-8)Hz_*")
    broad_band_files = collect_files(root_folder, files_name="FC_EEG_*")
    bands_files = [broad_band_files, delta_band_files, theta_band_files, alpha_band_files, beta_band_files]
    PSD_files = collect_files(root_folder, files_name="PSD_EEG_*")

    tau_stages = 6
    abeta_stages = 6
    freq_max = np.zeros((tau_stages, abeta_stages)) 
    aperiodic_exponent = np.zeros((tau_stages, abeta_stages)) 
    psd_max = np.zeros((tau_stages, abeta_stages, 5))
    psd_metrics = np.zeros((tau_stages, abeta_stages, 4)) #tau stage, abeta stage, metrics [alpha/broad; delta/broad]
    fc_metrics = np.zeros((tau_stages, abeta_stages, 5)) #tau stage, abeta stage, metrics [regular abs delta; regular abs theta; regular abs alpha; regular abs broad]
    psd_means = np.zeros((tau_stages, abeta_stages, 5)) #tau stage, abeta stage, band [broad (0.5-45)Hz, delta (0.5-4)Hz, theta (4-8)Hz, alpha (8-12)Hz, beta (12-30)Hz] 
    fc_reg_means = np.zeros((tau_stages, abeta_stages, 5, 2)) #tau stage, abeta stage, band, mean or mean absolute value
    fc_65x65_means = np.zeros((tau_stages, abeta_stages, 5, 2))
    fc_red_means = np.zeros((tau_stages, abeta_stages, 5, 2)) #tau stage, abeta stage, band, mean or mean absolute value
    fc_7x7_means = np.zeros((tau_stages, abeta_stages, 5, 2)) #tau stage, abeta stage, band, mean or mean absolute value

    electrodes_name_path = 'C:/Users/User/OneDrive - University of Pisa/Desktop/TVB_tutorials/Dati_Healthy/'

    # Channel names for real EEG (62 channels)
    eeg_real_data_names = np.load(electrodes_name_path + 'names_ch.npy')
    eeg_real_data_names = np.array(eeg_real_data_names)

    # Channel names for Brainstorm EEG (65 channels)
    eeg_brainstorm_65_names = np.array([
        'Fp1', 'Fp2', 'F4', 'F3', 'C3', 'C4', 'P4', 'P3', 'O2', 'O1',
        'F8', 'F7', 'T4', 'T3', 'T6', 'T5', 'Pz', 'Fz', 'IO1', 'IO2',
        'AF9', 'AF10', 'F9', 'F10', 'CB1', 'CB2', 'TP7', 'TP9', 'TP10',
        'TP8', 'Oz', 'Iz', 'PO4', 'PO3', 'CP5', 'CP6', 'CP1', 'CP2',
        'FT9', 'FT10', 'FC2', 'FC1', 'AF3', 'AF4', 'FC6', 'FC5', 'CPz',
        'P1', 'POz', 'P2', 'P6', 'C6', 'P5', 'C1', 'C2', 'C5', 'F2',
        'F6', 'F1', 'AF8', 'F5', 'AF7', 'Fpz', 'FCz', 'Cz', 'T8', 'T7',
        'P8', 'P7'
    ])
    common_names = extract_common_names()
    #print("Numero canali comuni:", len(common_names))
    #print(common_names)
    #print(len(common_names) == len(set(common_names)))
    indices_65 = [np.where(eeg_brainstorm_65_names == name)[0][0] for name in common_names]
    indices_62 = [np.where(eeg_real_data_names == name)[0][0] for name in common_names]

    for tau in range(tau_stages):
        for abeta in range(abeta_stages):
            for band in range(len(bands_files)):
                fc_matrix = file_extraction(bands_files[band], tau, abeta)
                """
                interested_nodes = [6, 7, 8, 9, 12, 13, 16]
                #7x7 submatrix with most altered areas
                fc_7x7 = fc_matrix['normal_size'][np.ix_(interested_nodes, interested_nodes)]
                fc_7x7_viz = 2 * (fc_7x7 - np.min(fc_7x7)) / (np.max(fc_7x7) - np.min(fc_7x7)) - 1
                # Plot corr
                fig = plt.figure(figsize=(12, 8)) 
                ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
                im = ax.imshow(fc_7x7_viz, cmap='bwr', vmin=-1, vmax=1, aspect='equal')
                ax.set_title(f"Crirtical AD areas abeta: {abeta}, tau: {tau}, band: {band}, {np.mean(fc_7x7):.4g}_({np.mean(np.abs(fc_7x7)):.4g})")
                cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
                plt.colorbar(im, cax=cax)
                """
                #fig_path = r"C:\Users\User\OneDrive - University of Pisa\Desktop\7x7_matrices"
                """
                fig_name = f"tau_{tau}_abeta_{abeta}_band_{band}7x7_PSD.png"
                os.makedirs(fig_path, exist_ok=True)
                plt.savefig(os.path.join(fig_path, fig_name), dpi=200)
                plt.close()
                fc_7x7_means[tau,abeta, band, 0] = np.mean(fc_7x7)
                fc_7x7_means[tau,abeta, band, 1] = np.mean(np.abs(fc_7x7))
                """
                fc_matrix_sub = fc_matrix['normal_size'][np.ix_(indices_65, indices_65)]
                # diagonale ~ 0
                #print("Diag mean:", np.mean(np.diag(fc_matrix_sub)))

                # simmetria
                #print("FC symmetry error:",
                #    np.mean(np.abs(fc_matrix_sub - fc_matrix_sub.T)))

                # range
                #print("FC min/max:",
                #    np.min(fc_matrix_sub), np.max(fc_matrix_sub))
                
                fc_65x65_means[tau,abeta, band, 0] = np.mean(fc_matrix['normal_size'])
                fc_65x65_means[tau,abeta, band, 1] = np.mean(np.abs(fc_matrix['normal_size']))
                fc_reg_means[tau,abeta, band, 0] = np.mean(fc_matrix_sub)
                fc_reg_means[tau,abeta, band, 1] = np.mean(np.abs(fc_matrix_sub))
                fc_red_means[tau,abeta, band, 0] = np.mean(fc_matrix['reduced_size'])
                fc_red_means[tau,abeta, band, 1] = np.mean(np.abs(fc_matrix['reduced_size']))

            psd_data = file_extraction(PSD_files, tau, abeta)
            psd_mean = psd_data['psd_mean']
            psd_all = psd_data['psd_all_channels']
            assert fc_matrix['normal_size'].shape[0] >= max(indices_65) + 1
            assert psd_all.shape[0] >= max(indices_65) + 1
            psd_all_ch = psd_all[indices_65, :]
            psd = np.mean(psd_all_ch, axis=0)
            freqs = psd_data['freqs']
            aperiodic_exponent[tau, abeta] = aperiodic_slope_manual(psd_all_ch, freqs)
            psd_means[tau, abeta, 0] = np.sum(psd)                               #broad
            psd_means[tau, abeta, 1] = np.sum(psd[(freqs < 4)])                  #delta
            psd_means[tau, abeta, 2] = np.sum(psd[(freqs >= 4) & (freqs < 8)])   #theta
            psd_means[tau, abeta, 3] = np.sum(psd[(freqs >= 8) & (freqs < 12)])  #alpha
            psd_means[tau, abeta, 4] = np.sum(psd[(freqs >= 12) & (freqs < 30)]) #beta
            psd_metrics[tau, abeta, 0] = psd_means[tau, abeta, 1]/psd_means[tau, abeta, 0] #Delta/broad
            psd_metrics[tau, abeta, 1] = psd_means[tau, abeta, 2]/psd_means[tau, abeta, 0] #Theta/Broad
            psd_metrics[tau, abeta, 2] = psd_means[tau, abeta, 3]/psd_means[tau, abeta, 0] #Alpha/broad
            psd_metrics[tau, abeta, 3] = psd_means[tau, abeta, 4]/psd_means[tau, abeta, 0] #Beta/Broad

            psd_max[tau, abeta, 0] = np.max(psd)
            psd_max[tau, abeta, 1] = np.max(psd[(freqs < 4)])
            psd_max[tau, abeta, 2] = np.max(psd[(freqs >= 4) & (freqs < 8)])
            psd_max[tau, abeta, 3] = np.max(psd[(freqs >= 8) & (freqs < 12)])
            psd_max[tau, abeta, 4] = np.max(psd[(freqs >= 12) & (freqs < 30)])
            idx_max = np.argmax(psd)
            freq_max[tau, abeta] = freqs[idx_max]
            #print("PSD total:", np.sum(psd))
            #print("Delta %:", psd_means[tau,abeta,1] / psd_means[tau,abeta,0])
            #print("Alpha %:", psd_means[tau,abeta,3] / psd_means[tau,abeta,0])
            #print("Peak freq:", freq_max[tau,abeta])

    fc_metrics[:,:,0] = fc_reg_means[:,:,0,1] #broad
    fc_metrics[:,:,1] = fc_reg_means[:,:,1,1] #delta
    fc_metrics[:,:,2] = fc_reg_means[:,:,2,1] #theta
    fc_metrics[:,:,3] = fc_reg_means[:,:,3,1] #alpha
    fc_metrics[:,:,4] = fc_reg_means[:,:,4,1] #beta

    #print(fc_metrics[:,:,0])

    #print("FC delta abs (tau increasing):",
    #  fc_metrics[:,0,0])

    #print("Alpha PSD ratio (tau increasing):",
    #    psd_metrics[:,0,0])
    #ratio = fc_65x65_means[:,:,:,1] / fc_reg_means[:,:,:,1]
    #print("Reduced/Normal FC ratio:", np.mean(ratio))
    print(aperiodic_exponent)
    return fc_7x7_means, fc_reg_means, fc_red_means, psd_means, freqs, psd_max, freq_max, psd_metrics, fc_metrics

fc_7x7_means, fc_reg_means, fc_red_means, psd_means, freqs, psd_max, freq_max, psd_metrics, fc_metrics = stage_metrics()

os.makedirs(root_folder_global_save, exist_ok=True)
output = os.path.join(root_folder_global_save, f"stage_metrics_del_25_01_(new_staging).npz")
print(fc_metrics.shape)
np.savez_compressed(
            output,
            fc_7x7_means=fc_7x7_means,
            fc_reg_means=fc_reg_means,
            fc_red_means=fc_red_means,
            psd_means=psd_means,
            freqs=freqs,
            psd_metrics=psd_metrics,
            fc_metrics=fc_metrics
        )

def file_extraction(files_group, tau_stage, abeta_stage):
    file_sel = files_group[(files_group["tau"] == tau_stage) & (files_group["abeta"] == abeta_stage)]
    file_path_sel = file_sel["file_path"]
    file_path_string = file_path_sel.iloc[0] 
    file_data = np.load(file_path_string)
    return file_data

def plot_function(x, title, labels=['broad','delta','theta','alpha','beta'], y = np.linspace(0,5,6), tau = False):
    plt.figure(figsize=(10, 12))
    for i in range(len(labels)):
        if tau == True:
            xlabel='Tau stages'
            plt.plot(y, x[:, 0, i], label=labels[i])
        else:
            xlabel='Abeta stages'
            plt.plot(y, x[0, :, i], label=labels[i])
    plt.legend()
    plt.grid()
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    if tau == False:
        title_figure = title + '_Abeta'
    else:
        title_figure = title + '_Tau'
    folder = root_folder_global_save
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, title_figure))
    plt.close()

def plot_heatmap(x, title, idx, folder = root_folder_global_save):
    fig = plt.figure(figsize=(8, 5)) 
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7]) 
    im = ax.imshow(x[:,:,idx], cmap='bwr', vmin=None, vmax=None, aspect='equal')
    title_text = title
    ax.set_title(title_text, fontsize=15)
    ax.set_ylabel('Tau Stage', fontsize=12)# Primo indice (righe)
    ax.set_xlabel('Abeta Stage', fontsize=12)  # Secondo indice (colonne)
    cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
    plt.colorbar(im, cax=cax)
    title_figure = title
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, title_figure))
    plt.close()

def plot_heatmaps():
    plot_heatmap(x=psd_means, title='broad PSD mean', idx = 0)
    plot_heatmap(x=psd_means, title='delta PSD mean', idx = 1)
    plot_heatmap(x=psd_means, title='theta PSD mean', idx = 2)
    plot_heatmap(x=psd_means, title='alpha PSD mean', idx = 3)
    plot_heatmap(x=psd_means, title='beta PSD mean', idx = 4)
    plot_heatmap(x=fc_reg_means[:, :, :, 0], title='broad FC mean reg', idx = 0) #tau stage, abeta stage, band, mean or mean absolute value
    plot_heatmap(x=fc_reg_means[:, :, :, 0], title='delta FC mean reg', idx = 1)
    plot_heatmap(x=fc_reg_means[:, :, :, 0], title='theta FC mean reg', idx = 2)
    plot_heatmap(x=fc_reg_means[:, :, :, 0], title='alpha FC mean reg', idx = 3)
    plot_heatmap(x=fc_reg_means[:, :, :, 0], title='beta FC mean reg', idx = 4)
    plot_heatmap(x=fc_red_means[:, :, :, 0], title='broad FC mean red', idx = 0)
    plot_heatmap(x=fc_red_means[:, :, :, 0], title='delta FC mean red', idx = 1)
    plot_heatmap(x=fc_red_means[:, :, :, 0], title='theta FC mean red', idx = 2)
    plot_heatmap(x=fc_red_means[:, :, :, 0], title='alpha FC mean red', idx = 3)
    plot_heatmap(x=fc_red_means[:, :, :, 0], title='beta FC mean red', idx = 4)
    plot_heatmap(x=fc_reg_means[:, :, :, 1], title='broad abs FC mean reg', idx = 0)
    plot_heatmap(x=fc_reg_means[:, :, :, 1], title='delta abs FC mean reg', idx = 1)
    plot_heatmap(x=fc_reg_means[:, :, :, 1], title='theta abs FC mean reg', idx = 2)
    plot_heatmap(x=fc_reg_means[:, :, :, 1], title='alpha abs FC mean reg', idx = 3)
    plot_heatmap(x=fc_reg_means[:, :, :, 1], title='beta abs FC mean reg', idx = 4)
    plot_heatmap(x=fc_red_means[:, :, :, 1], title='broad abs FC mean red', idx = 0)
    plot_heatmap(x=fc_red_means[:, :, :, 1], title='delta abs FC mean red', idx = 1)
    plot_heatmap(x=fc_red_means[:, :, :, 1], title='theta abs FC mean red', idx = 2)
    plot_heatmap(x=fc_red_means[:, :, :, 1], title='alpha abs FC mean red', idx = 3)
    plot_heatmap(x=fc_red_means[:, :, :, 1], title='beta abs FC mean red', idx = 4)

def plot_functions():
    plot_function(x = fc_reg_means[:, :, :, 0], title = 'EEG FC reg mean')
    plot_function(x = fc_red_means[:, :, :, 0], title = 'EEG FC red mean')
    plot_function(x = fc_reg_means[:, :, :, 1], title = 'EEG FC abs reg mean')
    plot_function(x = fc_red_means[:, :, :, 1], title = 'EEG FC abs red mean')
    plot_function(x = psd_max, title = 'Frequency amplitude')
    plot_function(x = psd_means, title = 'PSD mean')
    plot_function(x = fc_reg_means[:, :, :, 0], title = 'EEG FC reg mean',tau=True)
    plot_function(x = fc_red_means[:, :, :, 0], title = 'EEG FC red mean',tau=True)
    plot_function(x = fc_reg_means[:, :, :, 1], title = 'EEG FC abs reg mean',tau=True)
    plot_function(x = fc_red_means[:, :, :, 1], title = 'EEG FC abs red mean',tau=True)
    plot_function(x = psd_max, title = 'Frequency amplitude')
    plot_function(x = psd_means, title = 'PSD mean')
    plt.figure(figsize=(10, 12))
    stages = np.linspace(0,5,6)
    plt.plot(stages, freq_max[:, 0], label = 'tau stages')
    plt.plot(stages, freq_max[0, :], label = 'abeta stages')
    plt.legend()
    plt.grid()
    title_figure = 'Frequency of max amplitude'
    plt.title(title_figure, fontsize=18)
    folder = root_folder_global_save
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, title_figure))
    plt.close()

def plot_metrics():
    plot_heatmap(x=fc_metrics[:,:,:], title='delta abs FC mean reg', idx=0, folder = os.path.join(root_folder_global_save, 'metrics'))
    plot_heatmap(x=fc_metrics[:,:,:], title='theta abs FC mean reg', idx=1, folder = os.path.join(root_folder_global_save, 'metrics'))
    plot_heatmap(x=fc_metrics[:,:,:], title='alpha abs FC mean reg', idx=2, folder = os.path.join(root_folder_global_save, 'metrics'))
    plot_heatmap(x=psd_metrics[:,:,:], title='delta power over broad power', idx=0, folder = os.path.join(root_folder_global_save, 'metrics'))
    plot_heatmap(x=psd_metrics[:,:,:], title='alpha power over broad power', idx=1, folder = os.path.join(root_folder_global_save, 'metrics'))

#plot_functions()
#plot_heatmaps()
#plot_metrics()
"""
np.set_printoptions(precision=3)
print(f"broad\n{fc_7x7_means[:,:,0,1]}")
print(f"delta\n{fc_7x7_means[:,:,1,1]}")
print(f"theta\n{fc_7x7_means[:,:,2,1]}")
print(f"alpha\n{fc_7x7_means[:,:,3,1]}")
print(f"beta\n{fc_7x7_means[:,:,4,1]}")
"""
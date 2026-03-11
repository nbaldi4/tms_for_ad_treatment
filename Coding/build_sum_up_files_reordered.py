import pandas as pd
from pathlib import Path
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import global_variable_creation as gv
import sns
import argparse

from data_analysis_utils import real_vs_stages_PSD, real_vs_stages_FC, to_common_indices, macro_fc
from additional_utils import extract_common_names
from sklearn.feature_selection import mutual_info_regression
from specparam import SpectralGroupModel # Usiamo Group per matrici [n_regions, n_freqs]

import re
import pandas as pd
from pathlib import Path

def collect_files(root_folder_path, files_name="*.npz"):
    root_folder = Path(root_folder_path)
    data = []

    # Regex aggiornata: cerca tau e abeta seguiti da numeri
    # Cerca anche 'rs' (random seed) se presente nel percorso
    pattern = re.compile(r"tau_?(\d+).*?abeta_?(\d+)", re.IGNORECASE)
    rs_pattern = re.compile(r"_rs(\d+)", re.IGNORECASE)

    # rglob("*") cerca ricorsivamente. Usiamo il pattern passato come argomento.
    for file in root_folder.rglob(files_name):
        # Cerchiamo i valori sia nel nome del file che nel percorso (più sicuro)
        full_path_str = str(file)
        match = pattern.search(file.name)
        rs_match = rs_pattern.search(full_path_str)

        if match:
            tau_val = int(match.group(1))
            abeta_val = int(match.group(2))
            rs_val = int(rs_match.group(1)) if rs_match else 0

            data.append({
                "tau": tau_val,
                "abeta": abeta_val,
                "rs": rs_val,
                "file_name": file.name,
                "file_path": str(file.resolve())
            })

    return pd.DataFrame(data)

def file_extraction(files_group, tau_stage, abeta_stage, seed=None):
    # Filtriamo per tau e abeta
    mask = (files_group["tau"] == tau_stage) & (files_group["abeta"] == abeta_stage)
    
    # Se specifichi un seed (es. rs0), filtra anche quello
    if seed is not None:
        mask &= (files_group["rs"] == seed)
        
    file_sel = files_group[mask]

    if file_sel.empty:
        print(f"⚠️ Nessun file trovato per Tau:{tau_stage}, Abeta:{abeta_stage}")
        return None

    # Prendiamo il primo disponibile
    file_path_string = file_sel.iloc[0]["file_path"]
    return np.load(file_path_string)

def main(stages_folder='Random_seeds_16_02_M', signal = "EEG", seeds_number = 40):

    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", type=str, help="Signal to analyze")
    parser.add_argument("--stages_folder", type=str, help="folder containing data")
    parser.add_argument("--seeds_number", type=int, help="number of seeds being analyzed")

    args = parser.parse_args()
    if args.stages_folder:
        stages_folder = args.stages_folder
    
    if args.signal:
        signal = args.signal
    
    if args.seeds_number:
        seeds_number = args.seeds_number

    stages_folder = 'Random_seeds_16_02_M'
    root = os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data", stages_folder)
    df_broad = collect_files(root, files_name=f"FC_{signal}_tau_*")
    df_delta = collect_files(root, files_name=f"FC_{signal} delta_tau_*")
    df_theta = collect_files(root, files_name=f"FC_{signal} theta_tau_*")
    df_alpha = collect_files(root, files_name=f"FC_{signal} alpha_tau_*")
    df_beta = collect_files(root, files_name=f"FC_{signal} beta_tau_*")
    df_psd = collect_files(root, files_name=f"PSD_{signal}_tau_*")
    print(df_psd.columns)
    print(list(df_psd.columns))
    print(df_psd["file_name"])
    df_psd = df_psd[~df_psd["file_name"].str.contains("interested", case=False)]
    print(df_psd.columns)
    print(list(df_psd.columns))
    print(df_psd["file_name"])

    stages = np.arange(0,6)
    seeds = np.arange(0,seeds_number)
    bands = np.arange(0,5)
    fc_matrices = np.zeros((len(stages), len(stages), len(bands), len(seeds)), dtype=object)
    psd_arrays = np.zeros((len(stages), len(stages), len(seeds)), dtype=object)
    psd_freqs = np.zeros((len(stages), len(stages), len(seeds)), dtype=object)
    for abeta in stages:
        for tau in stages:
            for seed in seeds:
                data_broad = file_extraction(df_broad, tau_stage=tau, abeta_stage=abeta, seed=seed)
                data_delta = file_extraction(df_delta, tau_stage=tau, abeta_stage=abeta, seed=seed)
                data_theta = file_extraction(df_theta, tau_stage=tau, abeta_stage=abeta, seed=seed)
                data_alpha = file_extraction(df_alpha, tau_stage=tau, abeta_stage=abeta, seed=seed)
                data_beta = file_extraction(df_beta, tau_stage=tau, abeta_stage=abeta, seed=seed)
                data_psd = file_extraction(df_psd, tau_stage=tau, abeta_stage=abeta, seed=seed)
                datas = [data_broad, data_delta, data_theta, data_alpha, data_beta]
                
                for band in bands:
                    if datas[band]:
                        reg_FC = datas[band]['normal_size']
                        fc_matrices[abeta, tau, band, seed] = reg_FC
                        mean = np.mean(reg_FC)
                        abs_mean = np.mean(np.abs(reg_FC))
                        datas[band].close()

                psd_all = data_psd['psd_all_channels']
                psd_arrays[abeta, tau, seed] = np.mean(psd_all, axis=0)
                freqs = data_psd['freqs']
                psd_freqs[abeta, tau, seed] = freqs
    metadata_fc = {
        "stages": stages,
        "bands": ["broad", "delta", "theta", "alpha", "beta"],
        "seeds": seeds,
        "description": "Regular (65x65) size FC matrices. First index represent abeta stages, second tau stages, third bands, fouth seeds"
    }

    output_path_fc = os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Staging_highlights", stages_folder + "_stages_fc.npz")

    np.savez_compressed(
        output_path_fc, 
        fc_matrices=fc_matrices, 
        metadata=metadata_fc
    )

    # conversione a numpy liscio
    fc_matrices = np.stack(fc_matrices, axis=0) #numpy ndarray

    # nuovo salvataggio 
    #np.savez(os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Staging_highlights",'simulated_stages_fc_clean.npz'), 
    #         fc_matrices=fc_matrices,
    #         metadata_fc=metadata_fc
    #)

    metadata_psd = {
        "stages": stages,
        "seeds": seeds,
        "description": "PSD arrays (1x1025). First index represent abeta stages, second tau stages, third seeds. Frequencies are also present"
    }

    output_path_psd = os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Staging_highlights", stages_folder + "_stages_psd.npz")

    np.savez_compressed(
        output_path_psd, 
        psd_arrays=psd_arrays,
        psd_freqs=psd_freqs,
        metadata=metadata_psd
    )

    # conversione a numpy liscio
    psd_arrays = np.stack(psd_arrays, axis=0) #numpy ndarray
    psd_freqs = np.stack(psd_freqs, axis=0)

    # nuovo salvataggio 
    #np.savez(os.path.join(r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Staging_highlights",'simulated_stages_psd_clean.npz'), 
    #         psd_arrays=psd_arrays,
    #         psd_freqs=psd_freqs,
    #         metadata=metadata_psd
    #)


    loaded_psd_data = np.load(output_path_psd, allow_pickle=True)
    loaded_fc_data = np.load(output_path_fc, allow_pickle=True)
    fc_from_file = loaded_fc_data['fc_matrices']
    psd_from_file = loaded_psd_data['psd_arrays']
    print(fc_from_file.shape)
    print(psd_from_file.shape)
    print(fc_from_file[0,0,0,0].shape)
    print(psd_from_file[0,0,0].shape)
    # Verifichiamo prima la forma (se sono diverse, sono sicuramente diversi)
    if fc_from_file.shape == fc_matrices.shape:
        # Appiattiamo entrambi e usiamo un ciclo generator per controllare cella per cella
        # Questo evita l'ambiguità perché np.array_equal viene chiamato sulle singole matrici 65x65
        is_identical = all(np.array_equal(a, b) for a, b in zip(fc_from_file.flat, fc_matrices.flat))
        
        if is_identical:
            print("✅ Identical FC")
        else:
            print("❌ FC differiscono nel contenuto")
    else:
        print("❌ FC differiscono nella forma (Shape)")

    # TEST PSD: Nota che la chiave è 'psd_arrays', non 'psd_matrices' 
    # (controlla come l'hai salvata sopra!)
    if psd_from_file.shape == psd_arrays.shape:
        # Appiattiamo entrambi e usiamo un ciclo generator per controllare cella per cella
        # Questo evita l'ambiguità perché np.array_equal viene chiamato sulle singole matrici 65x65
        is_identical = all(np.array_equal(a, b) for a, b in zip(psd_from_file.flat, psd_arrays.flat))
        
        if is_identical:
            print("✅ Identical PSD")
        else:
            print("❌ PSD differiscono nel contenuto")
    else:
        print("❌ PSD differiscono nella forma (Shape)")

    """
    fc_A = np.ones((2,2))
    fc_A[0,0] = 5
    fc_A[1,0] = 3

    fc_B = np.ones((3,2))
    fc_B[0,0] = 5
    fc_B[1,0] = 4

    if fc_A.shape == fc_B.shape:
        # Appiattiamo entrambi e usiamo un ciclo generator per controllare cella per cella
        # Questo evita l'ambiguità perché np.array_equal viene chiamato sulle singole matrici 65x65
        is_identical = all(np.array_equal(a, b) for a, b in zip(fc_A.flat, fc_B.flat))
        
        if is_identical:
            print("✅ Identical FC")
        else:
            print("❌ FC differiscono nel contenuto")
    else:
        print("❌ FC differiscono nella forma (Shape)")

    # TEST PSD: Nota che la chiave è 'psd_arrays', non 'psd_matrices' 
    # (controlla come l'hai salvata sopra!)

    psd_A = np.ones(3)
    psd_A[0] = 1
    psd_A[1] = 3

    psd_B = np.ones(3)
    psd_B[0] = 5
    psd_B[1] = 3
    if psd_A.shape == psd_B.shape:
        # Appiattiamo entrambi e usiamo un ciclo generator per controllare cella per cella
        # Questo evita l'ambiguità perché np.array_equal viene chiamato sulle singole matrici 65x65
        is_identical = all(np.array_equal(a, b) for a, b in zip(psd_A.flat, psd_B.flat))
        
        if is_identical:
            print("✅ Identical PSD")
        else:
            print("❌ PSD differiscono nel contenuto")
    else:
        print("❌ PSD differiscono nella forma (Shape)")



    print('abeta 0, tau 4, banda theta, seed 3')
    print(fc_matrices[0, 4, 2, 3].shape, '; ', np.mean(np.abs(fc_matrices[0, 4, 2, 3])))
    print(psd_arrays[0, 4, 3].shape, '; ', np.max(psd_arrays[0, 4, 3]))
    print(psd_freqs[0, 4, 3].shape)
    print('abeta 1, tau 5, banda alpha, seed 12')
    print(fc_matrices[1, 5, 3, 12].shape, '; ', np.mean(np.abs(fc_matrices[1, 5, 3, 12])))
    print(psd_arrays[1, 5, 12].shape, '; ', np.max(psd_arrays[1, 5, 12]))
    print(psd_freqs[1, 5, 12].shape)
    print('abeta 2, tau 1, banda delta, seed 1')
    print(fc_matrices[2, 1, 1, 1].shape, '; ', np.mean(np.abs(fc_matrices[2, 1, 1, 1])))
    print(psd_arrays[2, 1, 1].shape, '; ', np.max(psd_arrays[2, 1, 1]))
    print(psd_freqs[2, 1, 1].shape)
    """
if __name__ == "__main__":
    main()


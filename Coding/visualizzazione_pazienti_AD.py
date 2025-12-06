import numpy as np
import matplotlib.pyplot as plt

# --- 1. DATI E CARICAMENTO ---
# Array di dati clinici
Abeta_patients = np.array([1559.0, np.nan, 338.0, 974.0, np.nan, 384.0, np.nan, 1239.0, 758.0,
                         np.nan, np.nan, 1742.0, np.nan, 527.0, np.nan, 702.0, 1170.0, np.nan, 
                         812.0, 1238.0, np.nan, np.nan, 1202.0, 702.0, np.nan, 692.0, 876.0,
                         1251.0, 490.0, np.nan, 1320.0, 1962.0, np.nan, 1374.0, np.nan, 1572.0,
                         685.0, 698.0, np.nan, 464.0])

tau_patients = np.array([40.0, np.nan, 128.2, 20.0, np.nan, 30.6, np.nan, 44.0, 125.0, np.nan,
                        np.nan, 53.0, np.nan, 34.0, np.nan, 81.0, 28.0, np.nan, 56.0, 30.0,
                        np.nan, np.nan, 25.0, 40.0, np.nan, 103.0, 158.0, 36.0, 103.0, np.nan,
                        39.4, 102.6, np.nan, 40.4, np.nan, 42.9, 156.9, 83.0, np.nan, 18.2])

# Caricamento dati FC (Assumendo che i percorsi siano corretti)
MCI_FC_data = np.load(r'C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\MCI_data\fc_mci_preview.npy')
MCI_PSD_data = np.load(r'C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\MCI_data\psd_mci_preview.npy')

# Array di coppie [Abeta, Tau]
patients_data = np.stack((Abeta_patients, tau_patients), axis=1)

# --- 2. DEFINIZIONE MASCHERE E CALCOLO ---

is_valid = np.isfinite(tau_patients) & np.isfinite(Abeta_patients)
cond_tau_raw = (tau_patients > 52)
cond_abeta_raw = (Abeta_patients < 813)
cond_AD_raw = (Abeta_patients < 644 + 0.25 * tau_patients)

# Maschere Esclusive
maschera_solo_tau = cond_tau_raw & (~cond_abeta_raw) & (~cond_AD_raw) & is_valid
maschera_solo_abeta = cond_abeta_raw & (~cond_tau_raw) & (~cond_AD_raw) & is_valid
# Maschera AD Standard
maschera_AD_standard = cond_AD_raw & is_valid

# Indici base e array filtrati
indici_solo_tau = np.where(maschera_solo_tau)[0]
tau_solo_tau = tau_patients[maschera_solo_tau]

indici_solo_abeta = np.where(maschera_solo_abeta)[0]
abeta_solo_abeta = Abeta_patients[maschera_solo_abeta]

indici_AD_standard = np.where(maschera_AD_standard)[0]
abeta_AD_standard = Abeta_patients[maschera_AD_standard]

# --- 3. RICERCA INDICI ESTREMI ---

# A. Max Tau (Solo Tau)
idx_tau = -1
if len(indici_solo_tau) > 0:
    idx_rel = np.argmin(tau_solo_tau)
    idx_tau = indici_solo_tau[idx_rel]

# B. Min Abeta (Solo Abeta)
idx_abeta = -1
if len(indici_solo_abeta) > 0:
    idx_rel = np.argmax(abeta_solo_abeta)
    idx_abeta = indici_solo_abeta[idx_rel]

# C. Min Abeta (AD Riemen-schneider)
idx_abeta_AD = -1
if len(indici_AD_standard) > 0:
    idx_rel_AD = np.argmax(abeta_AD_standard)
    idx_abeta_AD = indici_AD_standard[idx_rel_AD]

# --- 4. FUNZIONE DI PLOTTING ---

def plot_fc_matrix(patient_index, title, Abeta_val, tau_val):
    """Estrae la matrice FC, la normalizza e la plotta, includendo i valori clinici."""
    if patient_index == -1 or patient_index >= MCI_FC_data.shape[0]:
        print(f"Skipping plot for: {title} (Indice non valido o mancante).")
        return

    matrix = MCI_FC_data[patient_index, :, :]
    
    # Normalizzazione/Clipping
    matrix = np.clip(matrix, -0.999999, 0.999999)
    matrix = 2 * (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix)) - 1

    psd_matrix = MCI_PSD_data[patient_index, :]
    psd_matrix = psd_matrix[:180].astype(float)
    if np.max(psd_matrix) == 0:
        psd_matrix = psd_matrix + 1e-12
    psd_matrix /= np.max(psd_matrix)

    freqs = np.linspace(0, 45, len(psd_matrix))

    plt.figure(figsize=(10,4))
    plt.plot(freqs, psd_matrix, color='r', linewidth=2)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Normalized PSD")
    plt.title(f"{title} (Indice {patient_index})\n(Abeta={Abeta_val:.1f} mg/L, Tau={tau_val:.1f} mg/L)", fontsize=12)
    plt.legend()
    plt.grid()
    plt.show()
    
    # Plotting
    plt.figure(figsize=(7, 6))
    plt.title(f"{title} (Indice {patient_index})\n(Abeta={Abeta_val:.1f} mg/L, Tau={tau_val:.1f} mg/L)", fontsize=12)
    plt.imshow(matrix, cmap='bwr', vmin=-1, vmax=1)
    plt.colorbar(label='Connettività Funzionale (FC)')
    plt.xlabel('Regioni Cerebrali')
    plt.ylabel('Regioni Cerebrali')
    plt.show()

# --- 5. STAMPA E PLOT DEI RISULTATI ESTREMI ---

print("## 🥇 Pazienti con Valori Estremi Selezionati")
print("------------------------------------------")

# A. Max Tau nella categoria "Solo Tau"
if idx_tau != -1:
    Abeta_val, tau_val = patients_data[idx_tau]
    
    print(f"### Paziente SOLO Tau con Tau MAX (Indice: {idx_tau})")
    print(f"Valori: Abeta={Abeta_val:.1f}, Tau={tau_val:.1f}")
    
    plot_fc_matrix(idx_tau, "FC Paziente SOLO Tau con Tau MIN", Abeta_val, tau_val)
else:
    print("Nessun paziente SOLO Tau trovato.")

print("---")

# B. Min Abeta nella categoria "Solo Abeta"
if idx_abeta != -1:
    Abeta_val, tau_val = patients_data[idx_abeta]
    
    print(f"### Paziente SOLO Abeta con Abeta MIN (Indice: {idx_abeta})")
    print(f"Valori: Abeta={Abeta_val:.1f}, Tau={tau_val:.1f}")
    
    plot_fc_matrix(idx_abeta, "FC Paziente SOLO Abeta con Abeta MAX", Abeta_val, tau_val)
else:
    print("Nessun paziente SOLO Abeta trovato.")

print("---")

# C. Min Abeta nella categoria "AD Riemen-schneider"
if idx_abeta_AD != -1:
    Abeta_val, tau_val = patients_data[idx_abeta_AD]
    
    print(f"### Paziente AD Riemen-schneider con Abeta MAX (Indice: {idx_abeta_AD})")
    print(f"Valori: Abeta={Abeta_val:.1f}, Tau={tau_val:.1f}")
    
    plot_fc_matrix(idx_abeta_AD, "FC Paziente AD Riemen-schneider con Abeta MIN", Abeta_val, tau_val)
else:
    print("Nessun paziente AD Riemen-schneider trovato.")
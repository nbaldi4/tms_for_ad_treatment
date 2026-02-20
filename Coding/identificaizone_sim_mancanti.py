import os
from itertools import product

# 1. Configurazione percorsi
root_folder = r"C:\Users\User\OneDrive - University of Pisa\Desktop\tms_for_ad_treatment\Results_data\Prova_28_01"

# 2. Definiamo i range (se sono da 0 a 5, usa range(6); se sono da 0 a 6, usa range(7))
# Basandoci sul tuo messaggio, ipotizziamo range(6) per x, y, z, k, q
valori = range(6) 

# Generiamo tutte le combinazioni teoriche (xyzkq)
combinazioni_teoriche = list(product(valori, valori, valori, valori, valori))

# 3. Leggiamo le cartelle esistenti
cartelle_reali = os.listdir(root_folder)

# Creiamo un set delle combinazioni già completate estraendole dai nomi delle cartelle
completate = set()
for cartella in cartelle_reali:
    # Cerchiamo di estrarre i numeri dal nome tipo: a0_b_abeta0_tau0_J0_weights1
    try:
        # Questo split rompe la stringa nei punti chiave
        # a0 -> 0, abeta0 -> 0, tau0 -> 0, J0 -> 0, weights1 -> 1
        parti = cartella.split('_')
        x = int(parti[0].replace('a', ''))
        y = int(parti[2].replace('abeta', ''))
        z = int(parti[3].replace('tau', ''))
        k = int(parti[4].replace('J', ''))
        q = int(parti[5].replace('weights', ''))
        completate.add((x, y, z, k, q))
    except (IndexError, ValueError):
        # Ignora file o cartelle che non seguono il pattern
        continue

# 4. Calcolo delle mancanze
mancanti = [c for c in combinazioni_teoriche if c not in completate]

# 5. Output dei risultati
print(f"Combinazioni totali attese: {len(combinazioni_teoriche)}")
print(f"Combinazioni trovate: {len(completate)}")
print(f"Combinazioni mancanti: {len(mancanti)}")

if mancanti:
    print("\nPrime 1000 combinazioni mancanti (x, y, z, k, q):")
    for m in mancanti:
        print(f"a{m[0]}_b_abeta{m[1]}_tau{m[2]}_J{m[3]}_weights{m[4]}")
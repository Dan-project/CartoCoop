import matplotlib.pyplot as plt
import numpy as np

data_dict = {}
all_points = []

# Lire le fichier et charger les données
try:
    with open('semantic.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Carte"):
                continue
            parts = line.split(',')
            if len(parts) == 4:
                label = parts[0]
                coords = [float(parts[1]), float(parts[2]), float(parts[3])]
                data_dict.setdefault(label, []).append(coords)
                all_points.append(coords)
except FileNotFoundError:
    print("Erreur : Le fichier 'semantique.txt' n'a pas été trouvé.")
    exit()

if not all_points:
    print("Erreur : Aucun point valide trouvé dans le fichier.")
    exit()

all_points = np.array(all_points)

# Configuration du graphique 2D (Vue de haut)
fig, ax = plt.subplots(figsize=(10, 8))

# Filtre et affichage
LABELS_A_AFFICHER = {'road','vegetation','building'}  # ← modifie ici selon besoin

for label, points in data_dict.items():
    if label not in LABELS_A_AFFICHER:
        continue
    points = np.array(points)
    ax.scatter(points[:, 0], points[:, 1], label=label, s=0.5, alpha=0.6)

# Habillage du graphique
ax.set_xlabel('X (Monde)')
ax.set_ylabel('Y (Monde)')
ax.set_title('Visualisation 2D (Vue de haut) de la Carte Sémantique')

lgnd = ax.legend(loc="upper right")
for handle in lgnd.legendHandles:
    handle.set_sizes([20.0])

ax.set_aspect('equal', 'box')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
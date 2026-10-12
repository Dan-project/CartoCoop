import matplotlib.pyplot as plt
import numpy as np

data_dict = {}
all_points = []

#  Lire le fichier et charger les données
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
                
                if label not in data_dict:
                    data_dict[label] = []
                data_dict[label].append(coords)
                all_points.append(coords)
                
except FileNotFoundError:
    print("Erreur : Le fichier 'semantique.txt' n'a pas été trouvé.")
    exit()

if not all_points:
    print("Erreur : Aucun point valide trouvé dans le fichier.")
    exit()

# Convertir tous les points pour le calcul des limites globales
all_points = np.array(all_points)
X_all, Y_all = all_points[:, 0], all_points[:, 1]

#  Configuration du graphique 2D (Vue de haut)
fig, ax = plt.subplots(figsize=(10, 8))

# Affichage des points par classe sémantique (points minuscules)
for label, points in data_dict.items():
    points = np.array(points)
    # s=0.5 rend les points très petits, alpha=0.6 gère la superposition
    ax.scatter(points[:, 0], points[:, 1], label=label, s=0.5, alpha=0.6)

# Habillage du graphique
ax.set_xlabel('X (Monde)')
ax.set_ylabel('Y (Monde)')
ax.set_title('Visualisation 2D (Vue de haut) de la Carte Sémantique')

# Légende adaptative (les marqueurs dans la légende seront plus gros pour rester visibles)
lgnd = ax.legend(loc="upper right")
for handle in lgnd.legendHandles:
    handle.set_sizes([20.0])

# Garder des proportions égales entre X et Y pour ne pas déformer la carte
ax.set_aspect('equal', 'box')

plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
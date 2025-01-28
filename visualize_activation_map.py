import numpy as np
import matplotlib.pyplot as plt

# Charger le fichier activation_map.npy
activation_map_path = "./activation_map.npy"
activation_map = np.load(activation_map_path)

# Afficher les dimensions de la carte pour vérification
print(f"Dimensions de la carte d'activation : {activation_map.shape}")

# Visualiser la carte d'activation
plt.figure(figsize=(8, 6))
plt.title("Carte d'Activation")
plt.imshow(activation_map, cmap='viridis')  # Utiliser une palette adaptée
plt.colorbar()  # Ajouter une barre de couleurs
plt.axis('off')  # Désactiver les axes si nécessaire
plt.show()

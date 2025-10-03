# Saliency As A Schedule

## Introduction 
Implémentation de la méthode issue de la publication **Saliency As A Schedule: Intuitive Image Attribution** https://ieeexplore.ieee.org/document/10647374

Cette implémentation propose une approche d'attribution visuelle pour les réseaux de neurones convolutifs en utilisant une méthode d'optimisation basée sur un masque de saillance. L'objectif est de générer des cartes de chaleur qui expliquent quelles régions d'une image influencent le plus la prédiction du modèle.

## Prérequis

### Dépendances Python
```bash
pip install torch torchvision numpy opencv-python pillow matplotlib pytorch-grad-cam
```

### Versions recommandées
- Python 3.8+
- PyTorch 1.10+
- CUDA (optionnel, pour accélération GPU)

## Structure du projet

```
.
├── README.md                          # Ce fichier
├── gradcam.py                         # Implémentation GradCAM pour comparaison
├── visualize_activation_map.py        # Visualisation des cartes d'activation
├── sas.py                             # Implémentation principale (Saliency As A Schedule)
├── input.jpg                          # Image d'entrée (à fournir)
├── activation_map.npy                 # Carte d'activation générée par GradCAM
├── sas_saliency.jpg                   # Carte de saillance générée par SAS
└── sas_overlay.jpg                    # Superposition de la carte sur l'image
```

## Utilisation

### 1. GradCAM (méthode de référence)

GradCAM utilise les gradients pour générer des cartes d'activation basées sur les couches convolutives du réseau.

```bash
python gradcam.py
```

**Sorties:**
- `activation_map.npy` : Carte d'activation brute
- Affichage interactif de la visualisation

Pour visualiser la carte d'activation sauvegardée:
```bash
python visualize_activation_map.py
```

### 2. Saliency As A Schedule (méthode proposée)

La méthode SAS optimise un masque de saillance en utilisant une approche d'insertion/suppression avec un scheduler.

```bash
python sas.py
```

**Sorties:**
- `sas_saliency.jpg` : Carte de saillance en niveaux de gris
- `sas_overlay.jpg` : Superposition colorée sur l'image originale

## Configuration

Les paramètres de la méthode SAS peuvent être ajustés dans le dictionnaire `CONFIG` du fichier `sas.py`:

```python
CONFIG = {
    "image_path": "input.jpg",          # Chemin vers l'image d'entrée
    "target_class": 1,                  # Classe cible (None pour prédiction automatique)
    "num_iters": 100,                   # Nombre d'itérations d'optimisation
    "lr": 0.01,                         # Taux d'apprentissage
    "K": 20,                            # Nombre de masques dans le schedule
    "gaussian_kernel_size": 17,         # Taille du noyau de lissage gaussien
    "blur_ksize": 51,                   # Taille du noyau de flou pour l'arrière-plan
    "temp_factor": 0.3,                 # Facteur de température pour le pooling
    "tv_lambda": 0.08,                  # Poids de la régularisation Total Variation
}
```

### Paramètres clés

- **num_iters**: Plus le nombre d'itérations est élevé, plus la carte de saillance sera raffinée (mais calcul plus long)
- **K**: Nombre d'étapes dans le schedule d'insertion/suppression (affecte la granularité)
- **tv_lambda**: Contrôle la régularité spatiale de la carte de saillance (valeurs plus élevées = cartes plus lisses)
- **temp_factor**: Contrôle la netteté du pooling (valeurs plus faibles = transitions plus douces)

## Principe de fonctionnement

### GradCAM
1. Passe forward sur le réseau avec l'image d'entrée
2. Calcul des gradients par rapport à la couche convolutive ciblée
3. Pondération des cartes d'activation par les gradients
4. Génération de la carte de chaleur finale

### Saliency As A Schedule (SAS)
1. Initialisation d'un masque paramétrique `m`
2. Lissage spatial du masque via un filtre gaussien
3. Génération d'un schedule de K masques avec des seuils croissants
4. Pour chaque masque:
   - **Insertion**: Application du masque sur image floutée → mesure de la confiance
   - **Deletion**: Application du masque inverse → mesure de la perte de confiance
5. Optimisation du masque pour maximiser l'insertion et minimiser la deletion
6. Régularisation Total Variation pour assurer la cohérence spatiale

 être plus précises et mieux localisées que GradCAM, notamment pour des modèles comme VGG16 ou ResNet50.

## Références

- Publication originale: [Saliency As A Schedule: Intuitive Image Attribution](https://ieeexplore.ieee.org/document/10647374)
- GradCAM: [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)

## Licence

Ce code est fourni à des fins éducatives et de recherche.

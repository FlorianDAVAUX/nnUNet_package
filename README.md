# 🫁 LungSegmentation — nnUNetv2 Prediction Package

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Un **package Python** permettant d’exécuter facilement des prédictions de segmentation pulmonaire avec **nnUNetv2**,  
incluant le **téléchargement automatique des modèles**, la **préparation des fichiers d’entrée** et le **renommage des résultats**.  

Ce package peut être utilisé :
- soit **en ligne de commande** avec `nnunet_predict`
- soit **directement en Python** en important la fonction `nnunet_predict`.

---

## 🚀 Fonctionnalités

- 🔽 Téléchargement et extraction automatique des modèles depuis une URL.
- 🧩 Préparation du `dataset.json` pour la prédiction nnUNetv2.
- 🧠 Conversion automatique des images d’entrée vers le format `.nrrd`.
- ⚙️ Lancement direct de la prédiction avec gestion des folds.
- 🧹 Nettoyage des fichiers temporaires générés par nnUNetv2.
- 🏷️ Renommage automatique du fichier final de segmentation.

---

## 📦 Installation

```bash
git clone https://github.com/FlorianDAVAUX/nnUNet_package.git
cd nnUNet_package
pip install -e .
```

## 📦 Prérequis

Avant de lancer le script, assurez-vous d’avoir installé et configuré les éléments suivants :


```bash
git clone https://github.com/FlorianDAVAUX/nnUNet_package.git
cd nnUNet_package
pip install -e .
```

---


## ⚙️ Utilisation rapide

| Option         | Description | Exemple |
|----------------|-------------|---------|
| `--mode`       | Mode de prédiction (`Invivo` ou `Exvivo`) | `--mode Invivo` |
| `--structure`  | Structure à segmenter (`Parenchyma`, `Airways`, `Vascular`, `ParenchymaAirways`, `All`, `Lobes`) | `--structure Parenchyma` |
| `--input`      | Chemin vers l’image d’entrée (.nii, .nii.gz, .mha, .nrrd) | `--input ~/data/scan_patient.nrrd` |
| `--output`     | Dossier de sortie pour la prédiction (par défaut `prediction`) | `--output ~/predictions` |
| `--models_dir` | Chemin pour stocker ou chercher les modèles | `--models_dir ~/models` |
| `--name`       | Nom final du fichier de prédiction (sans extension) | `--name segmentation_parenchyme` |

---

### Exemple complet

```bash
nnunet_predict \
    --mode Invivo \
    --structure Parenchyma \
    --input ~/data/scan_patient.nrrd \
    --output ~/predictions \
    --models_dir ~/models \
    --name segmentation_parenchyme
```
---

## ⚙️ Utilisation depuis un script python 
```bash
from nnUNet_package.main import run_nnunet_prediction

run_nnunet_prediction(
    mode,
    structure,
    input_path,
    output,
    models_dir,
    name
)
```

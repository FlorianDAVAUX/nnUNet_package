

import os
import json
import shutil
import subprocess
import urllib.request
import argparse
import SimpleITK as sitk

from nnUNet_package.predict import nnunet_predict

# ============================================================#
#                   🔧 CONTEXTE GLOBAL                        #   
# ============================================================#
GLOBAL_CONTEXT = {
    "dataset_json_path": None,
    "dataset_labels": None,
}
# ============================================================#
#                       📦 UTILITAIRES                        #
# ============================================================#
def load_model_config(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def download_and_extract_model(model_url, model_name, default_dir=None):
    """Télécharge et extrait le modèle si absent."""
    model_path = os.path.join(default_dir, model_name)
    zip_path = os.path.join(default_dir, f"{model_name}.zip")

    if not os.path.exists(model_path):
        print(f"🔽 Téléchargement de {model_name} depuis {model_url}...")
        urllib.request.urlretrieve(model_url, zip_path)
        print("✅ Téléchargement terminé")

        print(f"📂 Extraction du modèle dans {model_path}...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_path)
        print(f"✅ Modèle extrait dans {model_path}")
    else:
        print(f"Le modèle '{model_name}' est déjà présent")

    # Cherche le dataset.json du modèle
    for root, _, files in os.walk(model_path):
        if "dataset.json" in files:
            GLOBAL_CONTEXT["dataset_json_path"] = os.path.join(root, "dataset.json")
            break

    if not GLOBAL_CONTEXT["dataset_json_path"]:
        raise FileNotFoundError("dataset.json introuvable dans le modèle.")

    # Charge les labels une seule fois
    with open(GLOBAL_CONTEXT["dataset_json_path"], "r") as f:
        dataset = json.load(f)
        raw_label_map = dataset.get("labels", {})
        GLOBAL_CONTEXT["dataset_labels"] = {int(v): k for k, v in raw_label_map.items() if int(v) > 0}


def edit_dataset_json_for_prediction(input_image):
    """
    Prépare le dataset.json pour la prédiction nnUNet.

    Args:
        input_image (str): Chemin de l'image d'entrée
    Returns:
        chemin du dataset.json modifié, chemin du dossier imagesTs
    """
    dataset_json_path = GLOBAL_CONTEXT.get("dataset_json_path")
    if not dataset_json_path:
        raise RuntimeError("dataset.json introuvable dans le contexte global.")

    with open(dataset_json_path, "r") as f:
        dataset = json.load(f)

    dataset.pop("training", None)
    dataset["numTraining"] = 0
    dataset["numTest"] = 1

    imagesTs_path = os.path.join(os.path.dirname(dataset_json_path), "imagesTs")
    os.makedirs(imagesTs_path, exist_ok=True)
    dst = os.path.join(imagesTs_path, "001_0000.nrrd")

    if os.path.exists(dst):
        os.remove(dst)

    ext = os.path.splitext(input_image)[1].lower()
    if ext == ".nrrd":
        os.symlink(os.path.abspath(input_image), dst)
    else:
        img = sitk.ReadImage(input_image)
        sitk.WriteImage(img, dst)

    dataset["test"] = [[f"./imagesTs/001_0000.nrrd"]]

    with open(dataset_json_path, "w") as f:
        json.dump(dataset, f, indent=4)

    return imagesTs_path


def rename_prediction_file(prediction_path, new_name):
    """
    Renomme le fichier de prédiction avec le nom donné par l'utilisateur.
    Exemple : 001.nrrd -> mon_nom.nrrd

    Args:
        prediction_path (str): Chemin du fichier de prédiction généré par nnUNet
        new_name (str): Nouveau nom pour le fichier de prédiction (sans extension)
    Returns:
        str: Nouveau chemin du fichier renommé
    """
    directory = os.path.dirname(prediction_path)
    new_path = os.path.join(directory, f"{new_name}.nrrd")

    if os.path.exists(prediction_path):
        os.rename(prediction_path, new_path)
        return new_path
    else:
        print("⚠️ Fichier de prédiction introuvable :", prediction_path)
        return prediction_path


def cleanup_prediction_files(output_path):
    """
    Supprime les fichiers temporaires générés par nnUNetv2.

    Args:
        output_path (str): Chemin du dossier de sortie contenant les fichiers à supprimer.
    """
    for fname in ["dataset.json", "plans.json", "predict_from_raw_data_args.json"]:
        fpath = os.path.join(output_path, fname)
        if os.path.exists(fpath):
            os.remove(fpath)



def run_nnunet_prediction(mode, structure, input_path, output_dir, models_dir, name="prediction"):
    """
    Exécute la prédiction nnUNetv2 avec les paramètres donnés.

    Args:
        mode (str): "Invivo" ou "Exvivo".
        structure (str): "Parenchyma", "Airways", "Vascular", "ParenchymaAirways", "All", "Lobes".
        input_path (str): Chemin vers l'image d'entrée (.nii, .mha, .nrrd...).
        output_dir (str): Dossier de sortie pour la prédiction.
        models_dir (str): Dossier pour stocker ou chercher les modèles.
        name (str): Nom du fichier de sortie final (sans extension).
    """

    # Vérifications et création des dossiers
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Chargement de la configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "models.json")
    config = load_model_config(config_path)
    model_info = config[mode][structure]

    # Téléchargement ou vérification du modèle
    download_and_extract_model(model_info["model_url"], model_info["model_name"], models_dir)

    # Préparation du dataset.json et du dossier imagesTs
    imagesTs_path = edit_dataset_json_for_prediction(input_path)

    # Construction du chemin vers le modèle entraîné
    model_path = os.path.join(models_dir, model_info["model_name"])
    first = next((d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))), None)
    model_path = os.path.join(model_path, first)
    second = next((d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))), None)
    model_path = os.path.join(model_path, second)

    folds = (model_info["fold"],)

    # Exécution de la prédiction
    nnunet_predict(i=imagesTs_path, o=output_dir, m=model_path, f=folds)

    # Renommage du fichier de sortie
    prediction_file = os.path.join(output_dir, "001.nrrd")
    segmentation_path = rename_prediction_file(prediction_file, name)

    # Nettoyage des fichiers inutiles
    cleanup_prediction_files(output_dir)

    print("✅ Prédiction terminée :", segmentation_path)
    return segmentation_path


def main():
    parser = argparse.ArgumentParser(description="Prédiction pulmonaire avec nnUNetv2")
    parser.add_argument("--mode", default="Invivo", choices=["Invivo", "Exvivo"])
    parser.add_argument("--structure", required=True, choices=["Parenchyma", "Airways", "Vascular", "ParenchymaAirways", "All", "Lobes"])
    parser.add_argument("--input", required=True, help="Image d'entrée (.nii, .mha, .nrrd...)")
    parser.add_argument("--output", default="prediction", help="Dossier de sortie")
    parser.add_argument("--models_dir", required=True, help="Dossier pour stocker les modèles")
    parser.add_argument("--name", default="prediction", help="Nom du fichier final")

    args = parser.parse_args()

    run_nnunet_prediction(
        mode=args.mode,
        structure=args.structure,
        input_path=args.input,
        output_dir=args.output,
        models_dir=args.models_dir,
        name=args.name
    )


if __name__ == "__main__":
    main()


# # ============================================================#
# #                       🚀 MAIN                               #
# # ============================================================#
# def main():
#     parser = argparse.ArgumentParser(description="Prédiction pulmonaire avec nnUNetv2")
#     parser.add_argument("--mode", default="Invivo", choices=["Invivo", "Exvivo"])
#     parser.add_argument("--structure", required=True, choices=["Parenchyma", "Airways", "Vascular", "ParenchymaAirways", "All", "Lobes"])
#     parser.add_argument("--input", required=True, help="Image d'entrée (.nii, .mha, .nrrd...)")
#     parser.add_argument("--output", default="prediction", help="Dossier de sortie")
#     parser.add_argument("--models_dir", required=True, help="Dossier pour stocker les modèles")
#     parser.add_argument("--name", default="prediction", help="Nom du fichier final")
#     args = parser.parse_args()

#     # Création du dossier models_dir et du dossier de sortie si nécessaire
#     if not os.path.isdir(args.models_dir):
#         os.makedirs(args.models_dir, exist_ok=True)
#     if not os.path.isdir(args.output):
#         os.makedirs(args.output, exist_ok=True)

#     # Chargement de la config
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     config = load_model_config(os.path.join(script_dir, "models.json"))
#     model_info = config[args.mode][args.structure]
#     download_and_extract_model(model_info["model_url"], model_info["model_name"], args.models_dir)
#     imagesTs_path = edit_dataset_json_for_prediction(args.input)

#     #Création du chemin du modèle
#     model_path = os.path.join(args.models_dir, model_info["model_name"])
#     first = next((d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))), None)
#     model_path = os.path.join(model_path, first)
#     second = next((d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))), None)
#     model_path = os.path.join(model_path, second)

#     #Créer un tuple avec les folds
#     folds = (model_info["fold"],)

#     # Lancement de la prédiction
#     nnunet_predict(
#         i=imagesTs_path,
#         o=args.output,
#         m=model_path,
#         f=folds
#     )

#     # Renommage et nettoyage
#     prediction_file = os.path.join(args.output, "001.nrrd")
#     segmentation_path = rename_prediction_file(prediction_file, args.name)
#     cleanup_prediction_files(args.output)

#     print("✅ Prédiction terminée :", segmentation_path)


# if __name__ == "__main__":
#     main()

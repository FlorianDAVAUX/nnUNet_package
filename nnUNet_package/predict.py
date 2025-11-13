import os
import shutil
import torch
import json
import urllib.request
from os.path import isdir
import SimpleITK as sitk
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.file_path_utilities import maybe_mkdir_p


# ============================================================#
#                       📦 CONTEXT                            #
# ============================================================#

GLOBAL_CONTEXT = {
    "dataset_json_path": None,
    "dataset_labels": None,
}

# ============================================================#
#                       📦 UTILITAIRES                        #
# ============================================================#

def nnunet_predict(i, o, m, f):

    disable_tta=False

    # Si un gpu est disponible, on l'utilise
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    predictor = nnUNetPredictor(tile_step_size=0.5,
                                use_gaussian=True,
                                use_mirroring=not disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=True)
    predictor.initialize_from_trained_model_folder(m, f)
    predictor.predict_from_files(i, o, save_probabilities=False,
                                 overwrite=True,
                                 num_processes_preprocessing=3,
                                 num_processes_segmentation_export=3,
                                 folder_with_segs_from_prev_stage=None,
                                 num_parts=1,
                                 part_id=0)


def load_model_config(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def download_and_extract_model(model_url, model_name, default_dir=None):
    """Télécharge et extrait le modèle si absent."""
    model_path = os.path.join(default_dir, model_name)
    zip_path = os.path.join(default_dir, f"{model_name}.zip")

    if not os.path.exists(model_path):
        print(f"Téléchargement de {model_name} depuis {model_url}...")
        urllib.request.urlretrieve(model_url, zip_path)
        print(f"Extraction du modèle dans {model_path}...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_path)
        print(f"Modèle extrait dans {model_path}")
    else:
        print(f"Le modèle '{model_name}' est déjà présent")
    
    # Suppression du zip
    if os.path.exists(zip_path):
        os.remove(zip_path)

    # Recherche du dataset.json avec la fonction utilitaire
    GLOBAL_CONTEXT["dataset_json_path"] = find_dataset_json(model_path)

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
        chemin du dossier imagesTs
    """
    dataset_json_path = GLOBAL_CONTEXT.get("dataset_json_path")
    if not dataset_json_path:
        raise RuntimeError("dataset.json introuvable dans le contexte global.")

    with open(dataset_json_path, "r") as f:
        dataset = json.load(f)

    # Suppression des infos training et mise à jour du test
    dataset.pop("training", None)
    dataset["numTraining"] = 0
    dataset["numTest"] = 1

    imagesTs_path = os.path.join(os.path.dirname(dataset_json_path), "imagesTs")
    os.makedirs(imagesTs_path, exist_ok=True)
    
    dst = os.path.join(imagesTs_path, "001_0000.nrrd")

    # Supprime tout lien ou fichier existant, même brisé
    if os.path.lexists(dst):
        os.remove(dst)

    ext = os.path.splitext(input_image)[1].lower()
    if ext == ".nrrd":
        # Crée le symlink uniquement pour un .nrrd
        shutil.copy(os.path.abspath(input_image), dst)
    else:
        # Convertit tout autre format en .nrrd
        img = sitk.ReadImage(input_image)
        sitk.WriteImage(img, dst)

    dataset["test"] = [[f"./imagesTs/001_0000.nrrd"]]

    # Écriture du dataset.json modifié
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
    else:
        print("Fichier de prédiction introuvable :", prediction_path)


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

def find_dataset_json(model_dir):
    """
    Recherche récursivement le fichier dataset.json dans un dossier de modèle nnUNet.

    Args:
        model_dir (str): Dossier racine du modèle téléchargé.

    Returns:
        str: Chemin complet vers le dataset.json s'il est trouvé.
    Raises:
        FileNotFoundError: Si aucun dataset.json n'est trouvé.
    """
    for root, _, files in os.walk(model_dir):
        if "dataset.json" in files:
            return os.path.join(root, "dataset.json")
    raise FileNotFoundError(f"Aucun dataset.json trouvé dans {model_dir}")
    

def run_nnunet_prediction(mode, structure, input_path, output_dir, models_dir, animal):
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
    model_info = config[animal][mode][structure]

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

    # Nettoyage des fichiers inutiles
    cleanup_prediction_files(output_dir)

    return prediction_file


import os
import json
import shutil
import subprocess
import urllib.request
import argparse
import SimpleITK as sitk
import slicer
import vtk

# ============================================================
# 🔧 CONTEXTE GLOBAL
# ============================================================
GLOBAL_CONTEXT = {
    "dataset_json_path": None,
    "dataset_labels": None,
}


# ============================================================
# 📦 UTILITAIRES
# ============================================================
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
        print(f"Le modèle '{model_name}' est déjà présent dans {model_path}.")

    # # 🔄 Mise à jour du contexte global
    # GLOBAL_CONTEXT["model_path"] = model_path

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

    return model_path


def edit_dataset_json_for_prediction(input_image, model_path):
    """Prépare le dataset.json pour la prédiction nnUNet."""
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

    return dataset_json_path, imagesTs_path


def run_nnunet_prediction(input_dir, output_path, dataset_id, configuration, fold):
    """Lance la prédiction nnUNetv2."""
    print("🚀 Lancement de la prédiction avec nnUNetv2...")

    command = [
        "nnUNetv2_predict",
        "-i", input_dir,
        "-o", output_path,
        "-d", dataset_id,
        "-c", configuration,
        "-f", str(fold)
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True
    )

    for line in process.stdout:
        print(line, end="")
    process.stdout.close()
    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError("❌ Erreur lors de la segmentation.")
    return os.path.join(output_path, "001.nrrd")

def rename_prediction_file(prediction_path, new_name):
    """
    Renomme le fichier de prédiction avec le nom donné par l'utilisateur.
    Exemple : 001.nrrd -> mon_nom.nrrd
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
    for fname in ["dataset.json", "plans.json", "predict_from_raw_data_args.json"]:
        fpath = os.path.join(output_path, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            print(f"🗑 Supprimé : {fpath}")


# ============================================================
# 🚀 MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Prédiction pulmonaire avec nnUNetv2")
    parser.add_argument("--mode", default="Invivo", choices=["Invivo", "Exvivo"])
    parser.add_argument("--structure", required=True, choices=["Parenchyma", "Airways", "Vascular", "ParenchymaAirways", "All", "Lobes"])
    parser.add_argument("--input", required=True, help="Image d'entrée (.nii, .mha, .nrrd...)")
    parser.add_argument("--output", default="prediction", help="Dossier de sortie")
    parser.add_argument("--models_dir", required=True, help="Dossier pour stocker les modèles")
    parser.add_argument("--name", default="prediction", help="Nom du fichier final")
    args = parser.parse_args()

    # Création du dossier models_dir si nécessaire
    if not os.path.isdir(args.models_dir):
        os.makedirs(args.models_dir, exist_ok=True)

    # Chargement de la config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_model_config(os.path.join(script_dir, "models.json"))

    model_info = config[args.mode][args.structure]
    model_path = download_and_extract_model(model_info["model_url"], model_info["model_name"], args.models_dir)
    _, imagesTs_path = edit_dataset_json_for_prediction(args.input, model_path)

    os.makedirs(args.output, exist_ok=True)
    os.environ["nnUNet_results"] = os.path.abspath(os.path.join(args.models_dir, model_info["model_name"]))

    prediction_file = run_nnunet_prediction(
        imagesTs_path,
        args.output,
        model_info["model_id"],
        model_info["configuration"],
        model_info["fold"]
    )

    segmentation_path = rename_prediction_file(prediction_file, args.name)
    cleanup_prediction_files(args.output)

    print("✅ Prédiction terminée :", segmentation_path)


if __name__ == "__main__":
    main()

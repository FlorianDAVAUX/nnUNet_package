#!/usr/bin/env python3
import os
import json
import shutil
import tempfile
import subprocess
import urllib.request
import argparse
import SimpleITK as sitk

def load_model_config(json_path):
    """Charge la configuration des modèles depuis un fichier JSON."""
    with open(json_path, "r") as f:
        return json.load(f)

def download_and_extract_model(model_url, model_name, default_dir=None):
    """
    Vérifie si le modèle est déjà téléchargé. Sinon, demande un chemin et le télécharge/extrait.
    """
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

    return model_path


def edit_dataset_json_for_prediction(input, model_path):
    """Prépare dataset.json pour la prédiction nnUNet."""
    dataset_json_path = None
    for root, dirs, files in os.walk(model_path):
        if "dataset.json" in files:
            dataset_json_path = os.path.join(root, "dataset.json")
            break
    if not dataset_json_path:
        raise FileNotFoundError("dataset.json introuvable dans le modèle.")

    with open(dataset_json_path, "r") as f:
        dataset = json.load(f)

    dataset.pop("training", None)
    dataset["numTraining"] = 0
    dataset["numTest"] = 1

    imagesTs_path = os.path.join(os.path.dirname(dataset_json_path), "imagesTs")
    os.makedirs(imagesTs_path, exist_ok=True)
    dst = os.path.join(imagesTs_path, "001_0000.nrrd")
    ext = os.path.splitext(input)[1].lower()

    if os.path.exists(dst):
        os.remove(dst)

    if ext == ".nrrd":
        os.symlink(os.path.abspath(input), dst)
    else:
        img = sitk.ReadImage(input)
        sitk.WriteImage(img, dst)

    dataset["test"] = [[f"./imagesTs/001_0000.nrrd"]]

    with open(dataset_json_path, "w") as f:
        json.dump(dataset, f, indent=4)

    return dataset_json_path, imagesTs_path


def run_nnunet_prediction(input_nrrd_path, output_path, dataset_id, configuration, fold):
    """Exécute nnUNetv2_predict en ligne de commande."""

    print("🚀 Lancement de la prédiction avec nnUNetv2...")
    print(f"📁 Dossier de sortie : {output_path}")

    command = [
        "nnUNetv2_predict",
        "-i", input_nrrd_path,
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
        print(line, end='')

    process.stdout.close()
    return_code = process.wait()

    if return_code != 0:
        print("❌ Erreur de segmentation")
    else:
        return os.path.join(output_path, "001.nrrd")
    

def cleanup_prediction_files(output_path):
        files_to_remove = [
            "dataset.json",
            "plans.json",
            "predict_from_raw_data_args.json"
        ]
        for fname in files_to_remove:
            fpath = os.path.join(output_path, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
                print(f"🗑 Supprimé : {fpath}")


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


def main():
    parser = argparse.ArgumentParser(description="Prédiction pulmonaire avec nnUNetv2")
    parser.add_argument("--mode", default="Invivo", choices=["Invivo", "Exvivo"], help="Mode Invivo ou Exvivo (par défaut Invivo)")
    parser.add_argument("--structure", required=True, choices=["Parenchyma", "Airways", "Vascular", "ParenchymaAirways", "All", "Lobes"], help="Structure à segmenter")
    parser.add_argument("--input", required=True, help="Chemin vers l'image d'entrée (.nii, .nii.gz, .mha, .nrrd)")
    parser.add_argument("--output", default="prediction", help="Dossier de sortie")
    parser.add_argument("--models_dir", required=True, help="Chemin pour stocker les modèles")
    parser.add_argument("--name", default="prediction", help="Nom du fichier de prédiction final (sans extension)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "models.json")
    config = load_model_config(json_path)

    if args.structure not in config[args.mode]:
        raise ValueError(f"La structure {args.structure} n'existe pas pour le mode {args.mode}. Choix possibles : {list(config[args.mode].keys())}")

    if args.models_dir:
        args.models_dir = os.path.expanduser(args.models_dir)
        if os.path.exists(args.models_dir):
            if not os.path.isdir(args.models_dir):
                print(f"❌ Le chemin spécifié pour --models_dir n'est pas un dossier : {args.models_dir}")
                exit(1)
        else:
            os.makedirs(args.models_dir, exist_ok=True)

    model_info = config[args.mode][args.structure]

    # Ici on passe le chemin models_dir à download_and_extract_model
    model_path = download_and_extract_model(model_info["model_url"], model_info["model_name"], default_dir=args.models_dir)
    
    _, imagesTs_path = edit_dataset_json_for_prediction(args.input, model_path)

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    results_dir = os.path.join(args.models_dir, model_info["model_name"])
    os.environ["nnUNet_results"] = os.path.abspath(results_dir)

    prediction_file = run_nnunet_prediction(
        imagesTs_path,
        args.output,
        dataset_id=model_info["model_id"],
        configuration=model_info["configuration"],
        fold=model_info["fold"]
    )

    cleanup_prediction_files(args.output)
    prediction_file = rename_prediction_file(prediction_file, args.name)

    print("✅ Prédiction terminée :", prediction_file)


if __name__ == "__main__":
    main()

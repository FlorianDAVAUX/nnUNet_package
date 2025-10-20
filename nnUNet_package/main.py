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

def download_and_extract_model(model_url, model_name):
    """Télécharge et extrait le modèle si nécessaire."""
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = temp_dir.name
    zip_path = os.path.join(temp_path, f"{model_name}.zip")
    extracted_path = os.path.join(temp_path, model_name)

    if not os.path.exists(zip_path):
        print(f"🔽 Téléchargement de {model_name} depuis {model_url}...")
        urllib.request.urlretrieve(model_url, zip_path)
        print("✅ Téléchargement terminé")

    if not os.path.exists(extracted_path):
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_path)
        print(f"✅ Modèle extrait dans {extracted_path}")

    return extracted_path, temp_dir  # temp_dir doit rester vivant

def prepare_nrrd_input(input_path, temp_dir=None):
    """Convertit l'entrée en NRRD si nécessaire."""
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".nrrd":
        return input_path
    else:
        img = sitk.ReadImage(input_path)
        out_path = os.path.join(temp_dir if temp_dir else os.getcwd(), "converted_input.nrrd")
        sitk.WriteImage(img, out_path)
        return out_path

def edit_dataset_json_for_prediction(model_path, input_nrrd_path):
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
    shutil.copyfile(input_nrrd_path, dst)
    dataset["test"] = [[f"./imagesTs/001_0000.nrrd"]]

    with open(dataset_json_path, "w") as f:
        json.dump(dataset, f, indent=4)

    return dataset_json_path, imagesTs_path

def run_nnunet_prediction(input_nrrd_path, output_path, dataset_id, configuration, fold):
    """Exécute nnUNetv2_predict en ligne de commande."""
    command = [
        "nnUNetv2_predict",
        "-i", input_nrrd_path,
        "-o", output_path,
        "-d", dataset_id,
        "-c", configuration,
        "-f", str(fold)
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='')
    process.wait()
    if process.returncode != 0:
        raise RuntimeError("Erreur lors de l'exécution de nnUNetv2_predict")
    return os.path.join(output_path, "001.nrrd")

def main():
    parser = argparse.ArgumentParser(description="Prédiction pulmonaire avec nnUNetv2")
    parser.add_argument("--mode", default="Invivo", choices=["Invivo", "Exvivo"], help="Mode Invivo ou Exvivo (par défaut Invivo)")
    parser.add_argument("--structure", required=True, help="Structure à segmenter (Parenchyma, Airways, etc.)")
    parser.add_argument("--input", required=True, help="Chemin vers l'image d'entrée (.nii, .nii.gz, .mha, .nrrd)")
    parser.add_argument("--output", default="predictions", help="Dossier de sortie")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "models.json")
    config = load_model_config(json_path)

    if args.structure not in config[args.mode]:
        raise ValueError(f"La structure {args.structure} n'existe pas pour le mode {args.mode}. Choix possibles : {list(config[args.mode].keys())}")

    model_info = config[args.mode][args.structure]

    model_path, temp_dir = download_and_extract_model(model_info["model_url"], model_info["model_name"])
    input_nrrd = prepare_nrrd_input(args.input, temp_dir=temp_dir.name)
    edit_dataset_json_for_prediction(model_path, input_nrrd)

    os.makedirs(args.output, exist_ok=True)
    prediction_file = run_nnunet_prediction(
        input_nrrd, args.output,
        dataset_id=model_info["model_id"],
        configuration=model_info["configuration"],
        fold=model_info["fold"]
    )

    print("✅ Prédiction terminée :", prediction_file)

if __name__ == "__main__":
    main()

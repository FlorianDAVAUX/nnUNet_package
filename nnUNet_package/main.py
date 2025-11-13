import argparse
from .predict import run_nnunet_prediction, rename_prediction_file

def main():
    parser = argparse.ArgumentParser(description="Prédiction pulmonaire avec nnUNetv2")
    parser.add_argument("--mode", default="invivo", choices=["invivo", "exvivo", "axial"])
    parser.add_argument("--structure", required=True, choices=["parenchyma", "airways", "vascular", "parenchymaairways", "all", "lobes"])
    parser.add_argument("--input", required=True, help="Image d'entrée (.nii, .mha, .nrrd...)")
    parser.add_argument("--output", default="prediction", help="Dossier de sortie")
    parser.add_argument("--models_dir", required=True, help="Dossier pour stocker les modèles")
    parser.add_argument("--animal", default="rabbit", choices=["rabbit", "pig"])
    parser.add_argument("--name", default="prediction", help="Nom du fichier final")

    args = parser.parse_args()

    prediction_file = run_nnunet_prediction(
        mode=args.mode,
        structure=args.structure,
        input_path=args.input,
        output_dir=args.output,
        models_dir=args.models_dir,
        animal=args.animal,
    )

    rename_prediction_file(prediction_file, args.name)


if __name__ == "__main__":
    main()


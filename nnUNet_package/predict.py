import os
import shutil
import torch
import json
import urllib.request
import SimpleITK as sitk
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


############################################### GLOBAL CONTEXT FOR DATASET.JSON AND LABELS ##############################################

GLOBAL_CONTEXT = {
    "dataset_json_path": None,
    "dataset_labels": None,
}

############################################################### FUNCTIONS ###############################################################

def nnunet_predict(input_path, output_path, model_path, fold_id):
    """
    nnUNetv2 prediction function.

    Args:
        input_path (str): Path to the input image folder.
        output_path (str): Path to the output folder for results.
        model_path (str): Folder in which the trained model is. Must have subfolders fold_X for the different trainde folds.
        fold_id (str): Specify the folds of the trained model that should be used for prediction. Default: (0, 1, 2, 3, 4).

    Returns:
        None
    """
    disable_tta=False

    # If there is a GPU available, we use it
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Create a predictor object
    predictor = nnUNetPredictor(tile_step_size=0.5,
                                use_gaussian=True,
                                use_mirroring=not disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=True)
    
    # Initialize and run prediction
    predictor.initialize_from_trained_model_folder(model_path, fold_id)

    # Run prediction
    predictor.predict_from_files(input_path, output_path,
                                 save_probabilities=False,
                                 overwrite=True,
                                 num_processes_preprocessing=3,
                                 num_processes_segmentation_export=3,
                                 folder_with_segs_from_prev_stage=None,
                                 num_parts=1,
                                 part_id=0)


def load_model_config(json_path):
    """
    Load the json configuration file from the downloaded models.

    Args:
        json_path (str): Path to the JSON configuration file.

    Returns:
        dict: Content of the JSON configuration file as a Python dictionary.
    """
    with open(json_path, "r") as f:
        return json.load(f)


def download_and_extract_model(model_url, model_name, default_dir):
    """
    Download and extract the pretrained model if it does not already exist.

    Args:
        model_url (str): URL to download the model from.
        model_name (str): Name of the model directory.
        default_dir (str, optional): Directory to store the model. Defaults to current working directory.

    Returns:
        None
    """
    model_path = os.path.join(default_dir, model_name)
    zip_path = os.path.join(default_dir, f"{model_name}.zip")

    if not os.path.exists(model_path):
        print(f"Downloading model '{model_name}' from {model_url}...")
        urllib.request.urlretrieve(model_url, zip_path)
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_path)
        print(f"Model extracted to {model_path}")
    else:
        print(f"The model '{model_name}' is already present")
    
    # Remove the zip file
    if os.path.exists(zip_path):
        os.remove(zip_path)

    # Search for dataset.json using the utility function
    GLOBAL_CONTEXT["dataset_json_path"] = find_dataset_json(model_path)

    # Load labels only once
    with open(GLOBAL_CONTEXT["dataset_json_path"], "r") as f:
        dataset = json.load(f)
        raw_label_map = dataset.get("labels", {})
        GLOBAL_CONTEXT["dataset_labels"] = {int(v): k for k, v in raw_label_map.items() if int(v) > 0}


def edit_dataset_json_for_prediction(input_image):
    """
    Modify the dataset.json file for prediction with a single image.

    Args:
        input_image (str): Path to the input image.

    Returns:
        str: Path to the created imagesTs directory.
    """
    dataset_json_path = GLOBAL_CONTEXT.get("dataset_json_path")
    if not dataset_json_path:
        raise RuntimeError("dataset.json not found in the global context.")

    with open(dataset_json_path, "r") as f:
        dataset = json.load(f)

    # Delete training info and set test info
    dataset.pop("training", None)
    dataset["numTraining"] = 0
    dataset["numTest"] = 1

    # Create imagesTs directory, which is required by nnUNetv2 for the inference
    imagesTs_path = os.path.join(os.path.dirname(dataset_json_path), "imagesTs")
    os.makedirs(imagesTs_path, exist_ok=True)
    
    dst = os.path.join(imagesTs_path, "001_0000.nrrd")

    # Remove any existing link or file, even if broken
    if os.path.lexists(dst):
        os.remove(dst)

    ext = os.path.splitext(input_image)[1].lower()
    if ext == ".nrrd":
        # Create the symlink only for a .nrrd
        shutil.copy(os.path.abspath(input_image), dst)
    else:
        # Convert any other format to .nrrd
        img = sitk.ReadImage(input_image)
        sitk.WriteImage(img, dst)

    dataset["test"] = [[f"./imagesTs/001_0000.nrrd"]]

    # Write the modified dataset.json
    with open(dataset_json_path, "w") as f:
        json.dump(dataset, f, indent=4)

    return imagesTs_path


def rename_prediction_file(prediction_path, new_name):
    """
    Rename the prediction file from the nnUNetv2 output.

    Args:
        predicted_path (str): Path to the prediction file from nnUNetv2.
        new_name (str): New name for the prediction file (without extension).

    Returns:
        None
    """
    directory = os.path.dirname(prediction_path)
    new_path = os.path.join(directory, f"{new_name}.nrrd")

    if os.path.exists(prediction_path):
        os.rename(prediction_path, new_path)
    else:
        print("Prediction file not found:", prediction_path)


def cleanup_prediction_files(output_path):
    """
    Clean up the prediction files generated by nnUNetv2.

    Removes the following files from the output directory:
    - dataset.json
    - plans.json
    - predict_from_raw_data_args.json

    Args:
        output_path (str): Path to the directory containing the prediction files.

    Returns:
        None
    """
    for fname in ["dataset.json", "plans.json", "predict_from_raw_data_args.json"]:
        fpath = os.path.join(output_path, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

def find_dataset_json(model_dir):
    """
    Search for a dataset.json file in the given model directory.

    Args:
        model_dir (str): Directory to search for the dataset.json file.

    Returns:
        str: Path to the dataset.json file if found, otherwise raises a FileNotFoundError.

    Raises:
        FileNotFoundError: If a dataset.json file is not found in the given directory.
    """
    for root, _, files in os.walk(model_dir):
        if "dataset.json" in files:
            return os.path.join(root, "dataset.json")
    raise FileNotFoundError(f"dataset.json not found in {model_dir}")
    

def run_nnunet_prediction(mode, structure, input_path, output_dir, models_dir, animal):
    """
    Runs the nnUNetv2 prediction script.

    Args:
        mode (str): Mode of prediction (`invivo`, `exvivo`, `axial`).
        structure (str): Structure to segment (`parenchyma`, `airways`, `vascular`, `parenchymaairways`, `all`, `lobes`).
        input_path (str): Path to the input image.
        output_dir (str): Path to the output directory for results.
        models_dir (str): Path to the directory where the models are stored.
        animal (str): Animal to segment (`rabbit`, `pig`).

    Returns:
        str: Path to the prediction file.

    Raises:
        FileNotFoundError: If a dataset.json file is not found in the given directory.
    """

    # Create directories if they do not exist for models and output
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Load model configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "models.json")
    config = load_model_config(config_path)
    model_info = config[animal][mode][structure]

    # Download or verify the model
    download_and_extract_model(model_info["model_url"], model_info["model_name"], models_dir)

    # Prepare dataset.json and imagesTs folder
    imagesTs_path = edit_dataset_json_for_prediction(input_path)

    # Construct the path to the trained model
    model_path = os.path.join(models_dir, model_info["model_name"])
    first = next((d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))), None)
    model_path = os.path.join(model_path, first)
    second = next((d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))), None)
    model_path = os.path.join(model_path, second)

    fold_id = (model_info["fold"],)

    # Run prediction
    nnunet_predict(imagesTs_path, output_dir, model_path, fold_id)

    # Rename output file
    prediction_file = os.path.join(output_dir, "001.nrrd")

    # Clean up unnecessary files
    cleanup_prediction_files(output_dir)

    return prediction_file
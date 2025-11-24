# LungSegmentation nnUNetv2 Prediction Script

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A Python script to easily run **nnUNetv2** lung segmentation predictions
on medical images. Automatic model download, file preparation for
prediction, and result renaming included.

------------------------------------------------------------------------

## Features

-   Automatic download and extraction of models from a URL.
-   Preparation of `dataset.json` for nnUNet prediction.
-   Conversion of input images to `.nrrd` if necessary.
-   Prediction execution with detailed logs.
-   Automatic cleanup of temporary files.
-   Automatic renaming of the final prediction file.

------------------------------------------------------------------------

## Requirements

Before running the script, make sure you have installed and configured
the following:

``` bash
git clone https://github.com/FlorianDAVAUX/nnUNet_package.git
cd nnUNet_package
pip install -e .
```

------------------------------------------------------------------------

## Usage

| Option         | Description | Example |
|----------------|-------------|---------|
| `--mode`       | Prediction mode (`Invivo` or `Exvivo`) | `--mode Invivo` |  
| `--structure`  | Structure to segment (`Parenchyma`, `Airways`, `Vascular`, `ParenchymaAirways`, `All`, `Lobes`) | `--structure Parenchyma` |
| `--input`      | Path to the input image (.nii, .nii.gz, .mha, .nrrd) | `--input ~/data/scan_patient.nrrd` |
| `--output`     | Output directory for the prediction (default: `prediction`) | `--output ~/predictions` |
| `--models_dir` | Path to store or search for models | `--models_dir ~/models` |
| `--name`       | Final name of the prediction file (without extension) | `--name segmentation_parenchyma` |


### Full Example

``` bash
nnunet_predict --mode Invivo --structure Parenchyma --input ~/data/scan_patient.nrrd --output ~/predictions --models_dir ~/models --name segmentation_parenchyma
```
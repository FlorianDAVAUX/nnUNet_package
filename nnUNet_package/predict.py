from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.file_path_utilities import maybe_mkdir_p
import torch
from os.path import isdir

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
"""Model download and path management utilities for pretrained CASCADE models.

Extracted from cascade2p/cascade.py -- logic preserved exactly.
"""

from __future__ import annotations

import glob
import os
import re
import zipfile
from urllib.request import urlopen

DEFAULT_MODEL_FOLDER = "Pretrained_models"
DEFAULT_INFO_FILE_LINK = (
    "https://raw.githubusercontent.com/PTRRupprecht/CascadeTorch/"
    "refs/heads/master/Pretrained_models/available_models_CascadeTorch.yaml"
)


def get_model_paths(model_path: str) -> dict[int, list[str]]:
    """Find all ``.pth`` model files in *model_path* and return as a dictionary.

    This is a helper function called by the prediction pipeline.

    Parameters
    ----------
    model_path : str
        Path to the directory containing saved model ``.pth`` files.

    Returns
    -------
    dict[int, list[str]]
        Dictionary with noise level (int) as keys. Each value is a sorted list
        of absolute paths to the ensemble model files for that noise level.

    Raises
    ------
    Exception
        If no ``.pth`` files are found in *model_path*.
    """
    all_models = glob.glob(os.path.join(model_path, "*.pth"))
    all_models = sorted(all_models)

    # Exception in case no model was found
    if len(all_models) == 0:
        m = 'No models (*.pth files) were found in the specified folder "{}".'.format(
            os.path.abspath(model_path)
        )
        raise Exception(m)

    # dictionary with key for noise level, entries are lists of models
    model_dict: dict[int, list[str]] = {}
    for mp in all_models:
        try:
            noise_level = int(re.findall(r"_NoiseLevel_(\d+)", mp)[0])
        except Exception:
            print("Error while processing the file with name: ", mp)
            raise
        # add model path to the model dictionary
        if noise_level not in model_dict:
            model_dict[noise_level] = []
        model_dict[noise_level].append(mp)

    return model_dict


def download_model(
    model_name: str,
    model_folder: str = DEFAULT_MODEL_FOLDER,
    info_file_link: str = DEFAULT_INFO_FILE_LINK,
    verbose: int = 1,
) -> None:
    """Download and unzip a pretrained model from the online repository.

    Parameters
    ----------
    model_name : str
        Name of the model, e.g. ``'Global_EXC_30Hz_smoothing25ms'``.
        To see available models, call with ``model_name='update_models'`` and
        check the downloaded ``available_models_CascadeTorch.yaml`` file.
    model_folder : str
        Directory where the model will be saved.
    info_file_link : str
        Direct download link to the YAML file listing available models.
    verbose : int
        If > 0, print status messages.
    """
    from cascade2p import config

    # Download the current yaml file with information about available models
    new_file = os.path.join(model_folder, "available_models_CascadeTorch.yaml")
    with urlopen(info_file_link) as response:
        text = response.read()

    with open(new_file, "wb") as f:
        f.write(text)

    # check if the specified model_name is present
    download_config = config.read_config(new_file)

    if model_name not in download_config.keys():
        if model_name == "update_models":
            print(
                "You can now check the updated available_models_CascadeTorch.yaml "
                "file for valid model names."
            )
            print("File location:", os.path.abspath(new_file))
            return

        raise Exception(
            'The specified model_name "{}" is not in the list of available models. '.format(
                model_name
            )
            + "Available models for download are: {}".format(
                list(download_config.keys())
            )
        )

    if verbose:
        print('Downloading and extracting new model "{}"...'.format(model_name))

    # download and save .zip file of model
    download_link = download_config[model_name]["Link"]
    with urlopen(download_link) as response:
        data = response.read()

    tmp_file = os.path.join(model_folder, "tmp_zipped_model.zip")
    with open(tmp_file, "wb") as f:
        f.write(data)

    # unzip the model and save in the corresponding folder
    with zipfile.ZipFile(tmp_file, "r") as zip_ref:
        zip_ref.extractall(path=os.path.join(model_folder, model_name))

    os.remove(tmp_file)

    if verbose:
        print(
            'Pretrained model was saved in folder "{}"'.format(
                os.path.abspath(os.path.join(model_folder, model_name))
            )
        )

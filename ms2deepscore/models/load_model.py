from pathlib import Path
from typing import Union
import torch
from ms2deepscore.models.SiameseSpectralModel import SiameseSpectralModel


def load_model(filename: Union[str, Path]) -> SiameseSpectralModel:
    """
    Load a MS2DeepScore model (SiameseModel) from file.

    For example:

    .. code-block:: python

        from ms2deepscore.models import load_model
        model = load_model("model_file_xyz.hdf5")

    Parameters
    ----------
    filename
        Filename. Expecting saved SiameseModel.

    """
    model_settings = torch.load(filename)

    # Extract model parameters from the checkpoint
    model_params = model_settings['model_params']

    # Instantiate the SiameseSpectralModel with the loaded parameters
    model = SiameseSpectralModel(**model_params)
    model.load_state_dict(model_settings['model_state_dict'])
    model.eval()
    return model

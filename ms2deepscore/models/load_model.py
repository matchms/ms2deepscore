import json
from pathlib import Path
from typing import Union
import numpy as np
import torch
from ms2deepscore.__version__ import __version__
from ms2deepscore.models.EmbeddingEvaluatorModel import \
    EmbeddingEvaluationModel
from ms2deepscore.models.LinearEmbeddingEvaluation import LinearModel
from ms2deepscore.models.SiameseSpectralModel import SiameseSpectralModel
from ms2deepscore.SettingsMS2Deepscore import (SettingsEmbeddingEvaluator,
                                               SettingsMS2Deepscore)


def load_model(filename: Union[str, Path]) -> SiameseSpectralModel:
    """
    Load a MS2DeepScore model (SiameseModel) from file.

    For example:

    .. code-block:: python

        from ms2deepscore.models import load_model
        model = load_model("model_file_xyz.pt")

    Parameters
    ----------
    filename
        Filename. Expecting saved SiameseModel.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_settings = torch.load(filename, map_location=device)
    # Extract model parameters from the checkpoint
    model_params = model_settings["model_params"]

    # Instantiate the SiameseSpectralModel with the loaded parameters
    model = SiameseSpectralModel(settings=SettingsMS2Deepscore(**model_params, validate_settings=False))
    model.load_state_dict(model_settings["model_state_dict"])
    model.eval()
    return model


def load_embedding_evaluator(filename: Union[str, Path]) -> EmbeddingEvaluationModel:
    """
    Load a EmbeddingEvaluation model from file.

    For example:

    .. code-block:: python

        from ms2deepscore.models import load_embedding_evaluator
        model = load_embedding_evaluator("model_file_xyz.pt")

    Parameters
    ----------
    filename
        Filename. Expecting saved EmbeddingEvaluationModel.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_settings = torch.load(filename, map_location=device)
    if model_settings["version"] != __version__:
        print(f"The model version ({model_settings['version']}) does not match the version of MS2Deepscore "
              f"({__version__}), consider downloading a new model or changing the MS2Deepscore version")
    # Extract model parameters from the checkpoint
    model_params = model_settings["model_params"]

    # Instantiate the EmbeddingEvaluationModel with the loaded parameters
    model = EmbeddingEvaluationModel(settings=SettingsEmbeddingEvaluator(**model_params))
    model.load_state_dict(model_settings["model_state_dict"])
    model.eval()
    return model


def load_linear_model(filepath):
    """Load a LinearModel from json.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        model_params = json.load(f)

    loaded_model = LinearModel(model_params["degree"])
    loaded_model.model.coef_ = np.array(model_params['coef'])
    loaded_model.model.intercept_ = np.array(model_params['intercept'])
    loaded_model.poly._min_degree = model_params["min_degree"]
    loaded_model.poly._max_degree = model_params["max_degree"]
    loaded_model.poly._n_out_full = model_params["_n_out_full"]
    loaded_model.poly.n_output_features_ = model_params["n_output_features_"]
    return loaded_model

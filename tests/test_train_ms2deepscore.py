import os
from pathlib import Path
import numpy as np
import pytest
from ms2deepscore.models.load_model import \
    load_model as load_ms2deepscore_model
from ms2deepscore.models.SiameseSpectralModel import SiameseSpectralModel
from ms2deepscore.MS2DeepScore import MS2DeepScore
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore
from ms2deepscore.train_new_model.train_ms2deepscore import train_ms2ds_model, plot_history
from tests.create_test_spectra import pesticides_test_spectra, create_test_spectra

TEST_RESOURCES_PATH = Path(__file__).parent / 'resources'


def test_train_ms2ds_model(tmp_path):
    spectra = create_test_spectra(8)
    settings = SettingsMS2Deepscore(**{
        "mz_bin_width": 1.0,
        "epochs": 2,  # to speed up tests --> usually many more
        "base_dims": (100, 100),  # to speed up tests --> usually larger
        "embedding_dim": 50,  # to speed up tests --> usually larger
        "same_prob_bins": np.array([(-0.01, 0.5), (0.5, 1.0)]),
        "average_inchikey_sampling_count": 8,
        "batch_size": 8
        })
    _, history = train_ms2ds_model(spectra, pesticides_test_spectra(), tmp_path, settings)

    ms2ds_history_plot_file_name = os.path.join(tmp_path, settings.history_plot_file_name)
    plot_history(history["losses"], history["val_losses"], ms2ds_history_plot_file_name)

    # check if model is saved
    model_file_name = os.path.join(tmp_path, settings.model_file_name)
    assert os.path.isfile(model_file_name), "Expecte ms2ds model to be created and saved"

    ms2ds_model = load_ms2deepscore_model(model_file_name)
    assert isinstance(ms2ds_model, SiameseSpectralModel), "Expected a siamese model"

    ms2deepscore = MS2DeepScore(ms2ds_model)
    assert ms2deepscore.pair(spectra[0], spectra[0]) == 1

    result = ms2deepscore.matrix(spectra, spectra)
    assert result.shape == (len(spectra), len(spectra))


def test_too_little_spectra(tmp_path):
    """Test if the correct error is raised when there are less spectra than the batch size.

    See PR #155 for more details"""
    spectra = create_test_spectra(4)
    settings = SettingsMS2Deepscore(**{
        "epochs": 2,
        "average_inchikey_sampling_count": 40,
        "batch_size": 8
        })
    with pytest.raises(ValueError, match="The number of unique inchikeys must be larger than the batch size."):
        train_ms2ds_model(spectra, spectra, tmp_path, settings)

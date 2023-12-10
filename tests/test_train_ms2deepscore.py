import os
from pathlib import Path
import pytest
from matchms.importing import load_from_mgf
from ms2deepscore import BinnedSpectrum, SpectrumBinner
from ms2deepscore.models import SiameseModel
from ms2deepscore.models.load_model import \
    load_model as load_ms2deepscore_model
from ms2deepscore.train_new_model.SettingMS2Deepscore import \
    SettingsMS2Deepscore
from ms2deepscore.train_new_model.train_ms2deepscore import (bin_spectra,
                                                             train_ms2ds_model)
from ms2deepscore.utils import load_pickled_file
from tests.test_data_generators import create_test_spectra


TEST_RESOURCES_PATH = Path(__file__).parent / 'resources'


def test_bin_spectra(tmp_path):
    spectra = list(load_from_mgf(os.path.join(TEST_RESOURCES_PATH, "pesticides_processed.mgf")))
    binned_spectrums_training, binned_spectrums_val, spectrum_binner = bin_spectra(spectra, spectra, save_folder=tmp_path)

    assert isinstance(binned_spectrums_training, list)
    assert len(binned_spectrums_training) == len(spectra) == len(binned_spectrums_val)

    for binned_spectrum in binned_spectrums_training + binned_spectrums_val:
        assert isinstance(binned_spectrum, BinnedSpectrum)
    assert isinstance(spectrum_binner, SpectrumBinner)

    # check if binned spectra are saved
    binned_training_spectra_file_name = os.path.join(tmp_path, "binned_training_spectra.pickle")
    assert os.path.isfile(binned_training_spectra_file_name), "Expected binned training spectra to be created and saved"
    binned_training_spectra = load_pickled_file(binned_training_spectra_file_name)
    assert binned_training_spectra == binned_spectrums_training

    binned_validation_spectra_file_name = os.path.join(tmp_path, "binned_validation_spectra.pickle")
    assert os.path.isfile(binned_validation_spectra_file_name), "Expected binned validation spectra to be created and saved"

    spectrum_binner = os.path.join(tmp_path, "spectrum_binner.pickle")
    assert os.path.isfile(
        spectrum_binner), "Expected spectrum binner to be created and saved"


def test_train_ms2ds_model(tmp_path):
    spectra = create_test_spectra(8)
    settings = SettingsMS2Deepscore({"epochs": 2,
                                     "average_pairs_per_bin": 2,
                                     "batch_size": 8})
    train_ms2ds_model(spectra, spectra, tmp_path, settings)

    # check if model is saved
    model_file_name = os.path.join(tmp_path, settings.model_directory_name, settings.model_file_name)
    assert os.path.isfile(model_file_name), "Expecte ms2ds model to be created and saved"
    ms2ds_model = load_ms2deepscore_model(model_file_name)
    assert isinstance(ms2ds_model, SiameseModel), "Expected a siamese model"


def test_too_little_spectra(tmp_path):
    """Test if the correct error is raised when there are less spectra than the batch size.

    See PR #155 for more details"""
    spectra = create_test_spectra(4)
    settings = SettingsMS2Deepscore({"epochs": 2,
                                     "average_pairs_per_bin": 2,
                                     "batch_size": 8})
    with pytest.raises(ValueError, match="The number of unique inchikeys must be larger than the batch size."):
        train_ms2ds_model(spectra, spectra, tmp_path, settings)

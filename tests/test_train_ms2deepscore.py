import os
import pickle
from pathlib import Path
import pytest
from matchms.importing import load_from_mgf
from ms2deepscore import BinnedSpectrum, SpectrumBinner
from ms2deepscore.models import SiameseModel
from ms2deepscore.models.load_model import \
    load_model as load_ms2deepscore_model
from ms2deepscore.train_new_model.calculate_tanimoto_matrix import \
    calculate_tanimoto_scores_unique_inchikey
from ms2deepscore.train_new_model.train_ms2deepscore import (
    bin_spectra, train_ms2deepscore_wrapper, train_ms2ds_model)


TEST_RESOURCES_PATH = Path(__file__).parent / 'resources'


def load_pickled_file(filename: str):
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object


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


def test_train_wrapper_ms2ds_model(tmp_path):
    spectra = list(load_from_mgf(os.path.join(TEST_RESOURCES_PATH, "pesticides_processed.mgf")))
    train_ms2deepscore_wrapper(spectra, spectra, tmp_path, epochs=2)

    # check if model is saved
    model_file_name = os.path.join(tmp_path, "ms2deepscore_model.hdf5")
    assert os.path.isfile(model_file_name), "Expecte ms2ds model to be created and saved"
    ms2ds_model = load_ms2deepscore_model(model_file_name)
    assert isinstance(ms2ds_model, SiameseModel), "Expected a siamese model"

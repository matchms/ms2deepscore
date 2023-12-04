import os
import pickle
from matchms.exporting import save_as_mgf
from ms2deepscore.train_new_model.SettingMS2Deepscore import \
    SettingsMS2Deepscore
from ms2deepscore.wrapper_functions.training_wrapper_functions import \
    train_ms2deepscore_wrapper
from tests.create_test_spectra import pesticides_test_spectra


def load_pickled_file(filename: str):
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object


def test_train_wrapper_ms2ds_model(tmp_path):
    spectra = pesticides_test_spectra()
    positive_mode_spectra = [spectrum.set("ionmode", "positive") for spectrum in spectra[:20]]
    negative_mode_spectra = [spectrum.set("ionmode", "negative") for spectrum in spectra[20:]]

    save_as_mgf(positive_mode_spectra+negative_mode_spectra, filename=os.path.join(tmp_path, "clean_spectra.mgf"))
    settings = SettingsMS2Deepscore({"epochs": 2,
                                     "average_pairs_per_bin": 2,
                                     "ionisation_mode": "negative",
                                     "batch_size": 2})
    train_ms2deepscore_wrapper(tmp_path, "clean_spectra.mgf", settings, 5)
    assert os.path.isfile(os.path.join(tmp_path, "trained_models", settings.model_directory_name,
                                       "ms2deepscore_model.hdf5"))

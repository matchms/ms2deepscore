import os
import pickle
from pathlib import Path
from matchms.exporting import save_as_mgf
from matchms.importing import load_from_mgf
from ms2deepscore.models import SiameseModel
from ms2deepscore.models.load_model import \
    load_model as load_ms2deepscore_model
from ms2deepscore.train_new_model.SettingMS2Deepscore import \
    SettingsMS2Deepscore
from ms2deepscore.train_new_model.train_ms2deepscore import (bin_spectra,
                                                             train_ms2ds_model)
from ms2deepscore.train_new_model.training_wrapper_functions import \
    train_ms2deepscore_wrapper


TEST_RESOURCES_PATH = Path(__file__).parent / 'resources'


def load_pickled_file(filename: str):
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object


def test_train_wrapper_ms2ds_model(tmp_path):
    # dir = os.path.join(TEST_RESOURCES_PATH, "test_wrapper")
    spectra = list(load_from_mgf(os.path.join(TEST_RESOURCES_PATH, "pesticides_processed.mgf")))
    save_as_mgf(spectra, filename=os.path.join(tmp_path, "clean_spectra.mgf"))
    settings = SettingsMS2Deepscore({"epochs": 2,
                                     "average_pairs_per_bin": 2,
                                     "ionisation_mode": "negative"})
    train_ms2deepscore_wrapper(tmp_path, "clean_spectra.mgf", settings, 20)
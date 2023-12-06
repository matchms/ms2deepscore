import os
from pathlib import Path
from matchms.importing import load_from_mgf
from ms2deepscore.models import load_model
from ms2deepscore.MS2DeepScore import MS2DeepScore


TEST_RESOURCES_PATH = Path(__file__).parent / 'resources'


def pesticides_test_spectra():
    return list(load_from_mgf(os.path.join(TEST_RESOURCES_PATH, "pesticides_processed.mgf")))


def ms2deepscore_model_file_name():
    return os.path.join(TEST_RESOURCES_PATH, "testmodel.hdf5")


def ms2deepscore_model():
    return MS2DeepScore(load_model(ms2deepscore_model_file_name()))

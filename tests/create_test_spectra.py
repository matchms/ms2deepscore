import os
from pathlib import Path
from matchms.importing import load_from_mgf

TEST_RESOURCES_PATH = Path(__file__).parent / 'resources'


def pesticides_test_spectra():
    return list(load_from_mgf(os.path.join(TEST_RESOURCES_PATH, "pesticides_processed.mgf")))


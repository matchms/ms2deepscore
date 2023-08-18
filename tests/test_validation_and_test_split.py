from unittest.mock import patch
import numpy as np
import pytest
from matchms import Spectrum
from ms2deepscore.train_new_model.validation_and_test_split import (
    select_spectra_belonging_to_inchikey, select_unique_inchikeys,
    split_spectra_in_random_inchikey_sets)


@pytest.fixture
def sample_spectra():
    return [
        Spectrum(mz=np.array([100.1]), intensities=np.array([0.9]),
                 metadata={"inchikey": 14 * "A"}),
        Spectrum(mz=np.array([100.1]), intensities=np.array([0.9]),
                 metadata={"inchikey": 14 * "B"}),
        Spectrum(mz=np.array([100.1]), intensities=np.array([0.9]),
                 metadata={"inchikey": 14 * "B"}),
        Spectrum(mz=np.array([100.1]), intensities=np.array([0.9]),
                 metadata={"inchikey": 14 * "C"}),
    ]


def test_select_unique_inchikeys(sample_spectra):
    result = select_unique_inchikeys(sample_spectra)
    assert result == [14 * "A", 14 * "B", 14 * "C"]


def test_select_spectra_belonging_to_inchikey(sample_spectra):
    inchikeys = [14 * "A", 14 * "B"]
    result = select_spectra_belonging_to_inchikey(sample_spectra, inchikeys)
    assert len(result) == 3
    assert result[0].get("inchikey") == 14 * "A"


def test_split_spectra_in_random_inchikey_sets(sample_spectra):
    # TODO: this is still a dummy test mostly
    val, test, train = split_spectra_in_random_inchikey_sets(sample_spectra, 2)
    assert len(val) + len(test) + len(train) == 4

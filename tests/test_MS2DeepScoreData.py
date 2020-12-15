import numpy as np
import pytest
from matchms import Spectrum
from ms2deepscore import MS2DeepScoreData


def test_MS2DeepScoreData():
    """Test if default initalization works"""
    ms2ds_data = MS2DeepScoreData(100)
    assert ms2ds_data.mz_max == 1000.0, "Expected different default value."
    assert ms2ds_data.mz_min == 10.0, "Expected different default value."
    assert ms2ds_data.d_bins == 9.9, "Expected different calculated bin size."


def test_MS2DeepScoreData_set_min_max():
    """Test if other limits work well"""
    ms2ds_data = MS2DeepScoreData(100, mz_min=0.0, mz_max=100.0)
    assert ms2ds_data.mz_max == 100.0, "Expected different default value."
    assert ms2ds_data.mz_min == 0.0, "Expected different default value."
    assert ms2ds_data.d_bins == 1.0, "Expected different calculated bin size."


def test_MS2DeepScoreData_create_binned_spectrums():
    """Test if create binned spectrums method works."""
    ms2ds_data = MS2DeepScoreData(100, mz_min=0.0, mz_max=100.0)
    spectrum_1 = Spectrum(mz=np.array([10, 50, 100.]),
                          intensities=np.array([0.7, 0.2, 0.1]),
                          metadata={'inchikey': "test_inchikey_01"})
    spectrum_2 = Spectrum(mz=np.array([10, 40, 90.]),
                          intensities=np.array([0.4, 0.2, 0.1]),
                          metadata={'inchikey': "test_inchikey_02"})

    ms2ds_data.create_binned_spectrums([spectrum_1, spectrum_2])
    assert ms2ds_data.known_bins == [10, 40, 50, 90, 100], "Expected different known bins."
    assert len(ms2ds_data.spectrums_binned) == 2, "Expected 2 binned spectrums."
    assert ms2ds_data.spectrums_binned[0] == {0: 0.7, 2: 0.2, 4: 0.1}, "Expected different binned spectrum."
    assert np.all(ms2ds_data.inchikeys_all == np.array(["test_inchikey_01", "test_inchikey_02"])), \
        "Expected different inchikeys in array."


def test_MS2DeepScoreData_create_binned_spectrums_missing_inchikey():
    """Test if create binned spectrums method works with missing inchikey."""
    ms2ds_data = MS2DeepScoreData(100, mz_min=0.0, mz_max=100.0)
    spectrum_1 = Spectrum(mz=np.array([10, 50, 100.]),
                          intensities=np.array([0.7, 0.2, 0.1]),
                          metadata={'inchikey': "test_inchikey_01"})
    spectrum_2 = Spectrum(mz=np.array([10, 40, 90.]),
                          intensities=np.array([0.4, 0.2, 0.1]),
                          metadata={})

    assert np.all(ms2ds_data.inchikeys_all == np.array(["test_inchikey_01", None])), \
        "Expected different inchikeys in array."


def test_MS2DeepScoreData_set_generator_parameters():
    """Test if set_generator_parameters methods works well."""
    ms2ds_data = MS2DeepScoreData(100, mz_min=0.0, mz_max=100.0)
    assert ms2ds_data.generator_args is None, "Settings should not yet be set."

    ms2ds_data.set_generator_parameters(batch_size=20, shuffle=False)
    generator_args = ms2ds_data.generator_args
    assert generator_args["batch_size"] == 20, "Expected different setting."
    assert generator_args["shuffle"] == False, "Expected different setting."
    assert generator_args["augment_peak_removal_intensity"] == 0.2, "Expected different setting."

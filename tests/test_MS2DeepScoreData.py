import numpy as np
import pytest
from matchms import Spectrum
from ms2deepscore import MS2DeepScoreData


def test_MS2DeepScoreData():
    """Test if default initalization works"""
    ms2ds_model = MS2DeepScoreData(100)
    assert ms2ds_model.mz_max == 1000.0, "Expected different default value."
    assert ms2ds_model.mz_min == 10.0, "Expected different default value."
    assert ms2ds_model.d_bins == 9.9, "Expected differnt calculated bin size."


def test_MS2DeepScoreData_set_min_max():
    """Test if other limits work well"""
    ms2ds_model = MS2DeepScoreData(100, mz_min=0.0, mz_max=100.0)
    assert ms2ds_model.mz_max == 100.0, "Expected different default value."
    assert ms2ds_model.mz_min == 0.0, "Expected different default value."
    assert ms2ds_model.d_bins == 1.0, "Expected differnt calculated bin size."


def test_MS2DeepScoreData_create_binned_spectrums():
    """Test if create binned spectrums method works."""
    ms2ds_model = MS2DeepScoreData(100, mz_min=0.0, mz_max=100.0)
    spectrum_1 = Spectrum(mz=np.array([10, 50, 100.]),
                          intensities=np.array([0.7, 0.2, 0.1]),
                          metadata={'precursor_mz': 500.5})
    spectrum_2 = Spectrum(mz=np.array([10, 40, 90.]),
                          intensities=np.array([0.4, 0.2, 0.1]),
                          metadata={'precursor_mz': 500.11})

    ms2ds_model.create_binned_spectrums([spectrum_1, spectrum_2])
    assert ms2ds_model.known_bins == [10, 40, 50, 90, 100], "Expected different known bins."
    assert len(ms2ds_model.spectrums_binned) == 2, "Expected 2 binned spectrums."
    assert ms2ds_model.spectrums_binned[0] == {0: 0.7, 2: 0.2, 4: 0.1}, "Expected differnt binned spectrum."


def test_MS2DeepScoreData_set_generator_parameters():
    """Test if set_generator_parameters methods works well."""
    ms2ds_model = MS2DeepScoreData(100, mz_min=0.0, mz_max=100.0)
    assert ms2ds_model.generator_args is None, "Settings should not yet be set."

    ms2ds_model.set_generator_parameters(batch_size=20, shuffle=False)
    generator_args = ms2ds_model.generator_args
    assert generator_args["batch_size"] == 20, "Expected different setting."
    assert generator_args["shuffle"] == False, "Expected different setting."
    assert generator_args["augment_peak_removal_intensity"] == 0.2, "Expected different setting."

import numpy as np
import pytest
from matchms import Spectrum
from ms2deepscore.spectrum_binning_fixed import (bin_number_array_fixed,
                                                 create_peak_list_fixed,
                                                 set_d_bins_fixed,
                                                 unique_peaks_fixed)


def test_create_peak_list_fixed():
    mz = np.array([10, 20, 21, 30, 40], dtype="float")
    intensities = np.array([1, 1, 1, 1, 0.5], dtype="float")
    spectrum = Spectrum(mz=mz, intensities=intensities)
    class_values  = {0:0, 10:1, 11:2, 20:3, 30:4}
    peak_lists, missing_fractions = create_peak_list_fixed([spectrum, spectrum],
                                       class_values, d_bins=1,
                                       mz_max=1000.0, mz_min=10.0, peak_scaling=1.0)

    assert peak_lists[0] == peak_lists[1], "lists should be the same for identical input"
    assert peak_lists[0] == [(0, 1.0), (1, 1.0),
                             (2, 1.0), (3, 1.0),
                             (4, 0.5)]
    assert missing_fractions == [0.0, 0.0], "Expected different missing fractions"


def test_create_peak_list_fixed_missing_fraction():
    """Test with unknown peaks --> missing_fraction"""
    mz = np.array([10, 20, 25, 30, 40], dtype="float")
    intensities = np.array([1, 1, 1, 1, 0.5], dtype="float")
    spectrum = Spectrum(mz=mz, intensities=intensities)
    class_values  = {0:0, 10:1, 11:2, 20:3, 30:4}
    peak_lists, missing_fractions = create_peak_list_fixed([spectrum, spectrum],
                                                           class_values, d_bins=1,
                                                           mz_max=1000.0, mz_min=10.0, peak_scaling=1.0)

    assert peak_lists[0] == peak_lists[1], "lists should be the same for identical input"
    assert peak_lists[0] == [(0, 1.0), (1, 1.0), (3, 1.0), (4, 0.5)]
    assert missing_fractions[0] == pytest.approx(2/9, 1e-8), "Expected different missing fractions"


def test_set_d_bins_fixed():
    d_bins = set_d_bins_fixed(1000, mz_min=10.0, mz_max=100.0)
    assert d_bins == 0.09, "Expected different result (0.09)."


def test_unique_peaks_fixed():
    mz = np.array([10, 20, 20.01, 20.1, 30, 40], dtype="float")
    intensities = np.array([0, 0.5, 0.1, 0.2, 0.2, 0.4], dtype="float")
    spectrum = Spectrum(mz=mz, intensities=intensities)

    class_values, unique_peaks = unique_peaks_fixed([spectrum, spectrum], d_bins=0.09,
                                                    mz_max=100.0, mz_min=10.0)
    assert class_values == {0: 0, 111: 1, 112: 2, 222: 3, 333: 4}
    assert unique_peaks == [0, 111, 112, 222, 333]


def test_unique_peaks_fixed_upper_mzmax():
    """Test if only peaks between mz_min and mz_max are taken."""
    mz = np.array([10, 20, 20.01, 20.1, 30, 40, 100.1, 110], dtype="float")
    intensities = np.array([0, 0.5, 0.1, 0.2, 0.2, 0.4, 0.5, 1.0], dtype="float")
    spectrum = Spectrum(mz=mz, intensities=intensities)

    class_values, unique_peaks = unique_peaks_fixed([spectrum, spectrum], d_bins=0.09,
                                                    mz_max=100.0, mz_min=10.0)
    assert class_values == {0: 0, 111: 1, 112: 2, 222: 3, 333: 4}
    assert unique_peaks == [0, 111, 112, 222, 333]

def test_bin_number_array_fixed():
    mz = np.array([5, 10, 20, 21, 30, 40, 100.1], dtype="float")
    intensities = np.array([0.2, 1, 1, 1, 1, 0.5, 1], dtype="float")
    spectrum = Spectrum(mz=mz, intensities=intensities)
    bins = bin_number_array_fixed(spectrum.peaks.mz, d_bins=0.1, mz_max=100.0, mz_min=10.0)
    assert np.all(bins == np.array([0, 100, 110, 200, 300]))

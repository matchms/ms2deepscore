import os
import numpy as np
import pytest
from matchms import Spectrum
from ms2deepscore.spectrum_binning_linear import (bin_number_array_linear,
                                                  create_peak_list_linear,
                                                  set_d_bins_linear,
                                                  unique_peaks_linear)


def test_create_peak_list_linear():
    mz = np.array([10, 20, 21, 30, 40], dtype="float")
    intensities = np.array([1, 1, 1, 1, 0.5], dtype="float")
    spectrum = Spectrum(mz=mz, intensities=intensities)
    class_values  = {0:0, 10:1, 11:2, 20:3, 30:4}
    peak_lists = create_peak_list_linear([spectrum, spectrum], class_values,
                                         min_bin_size=1.0, d_bins=0.0, mz_min=10.0)
    
    assert peak_lists[0] == peak_lists[1], "lists should be the same for identical input"
    assert peak_lists[0] == [(0, 1.0), (1, 1.0),
                             (2, 1.0), (3, 1.0),
                             (4, 0.5)]
    
    
def test_set_d_bins_linear():
    d_bins = set_d_bins_linear(1000, min_bin_size=0.01, mz_min=10.0, mz_max=100.0)
    assert d_bins == pytest.approx(0.00016016016, 1e-6), "Expected different result."
    
    
def test_unique_peaks_linear():
    mz = np.array([10, 20, 20.01, 20.1, 30, 40], dtype="float")
    intensities = np.array([0, 0.5, 0.1, 0.2, 0.2, 0.4], dtype="float")
    spectrum = Spectrum(mz=mz, intensities=intensities)
    
    class_values, unique_peaks = unique_peaks_linear([spectrum, spectrum], 
                                                     min_bin_size=0.01, d_bins=0.001, mz_min=10.0)
    assert class_values == {0: 0, 131: 1, 132: 2, 189: 3, 234: 4}
    assert unique_peaks == [0, 131, 132, 189, 234]
    
    
def test_bin_number_array_linear():
    mz = np.array([10, 20, 21, 30, 40], dtype="float")
    intensities = np.array([1, 1, 1, 1, 0.5], dtype="float")
    spectrum = Spectrum(mz=mz, intensities=intensities)
    bins = bin_number_array_linear(spectrum.peaks.mz, min_bin_size=0.01, d_bins=0.001, mz_min=10.0)
    assert np.all(bins == np.array([0, 131, 138, 189, 234]))

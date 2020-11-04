""" Functions to create binned vector from spectrum using fixed width bins.
"""
import numpy as np


def create_peak_list_fixed(spectrums, class_values, d_bins, mz_min=10.0, weight_power = 0.2):
    """Create list of (binned) peaks."""
    peak_lists = []

    for spectrum in spectrums:
        doc = bin_number_array_fixed(spectrum.peaks.mz, d_bins, mz_min=mz_min)
        weights = spectrum.peaks.intensities ** weight_power
        doc_bow = [class_values[x] for x in doc]
        peak_lists.append(list(zip(doc_bow, weights)))

    return peak_lists


def unique_peaks_fixed(spectrums, d_bins, mz_min):
    """Collect unique (binned) peaks."""
    unique_peaks = set()
    for spectrum in spectrums:
        for mz in bin_number_array_fixed(spectrum.peaks.mz, d_bins, mz_min):
            unique_peaks.add(mz)
    unique_peaks = sorted(unique_peaks)
    class_values = {}

    for i, item in enumerate(unique_peaks):
        class_values[item] = i

    return class_values, unique_peaks


def bin_size_fixed(number, d_bins):
    return d_bins * number


def bin_number_fixed(mz, d_bins):
    """Return binned position"""
    return int(mz/d_bins)


def bin_number_array_fixed(mz_array, d_bins, mz_min):
    """Return binned position"""
    assert np.all(mz_array >= mz_min), "Found peaks > mz_min."
    bins = mz_array/d_bins
    return (bins - bin_number_fixed(mz_min, d_bins)).astype(int)


def set_d_bins_fixed(number_of_bins, mz_min=10.0, mz_max=1000.0):
    return (mz_max - mz_min) / number_of_bins

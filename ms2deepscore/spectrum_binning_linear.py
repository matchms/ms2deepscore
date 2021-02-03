""" Functions to create binned vector from spectrum using linearly increasing width bins.
"""
def create_peak_list_linear(spectrums, class_values, 
                            min_bin_size, d_bins, mz_min=10.0):
    """Create list of (binned) peaks."""
    peak_lists = []

    for spectrum in spectrums:
        doc = bin_number_array_linear(spectrum.peaks.mz, min_bin_size, d_bins, mz_min=mz_min)
        weights = spectrum.peaks.intensities  # ** weight_power
        doc_bow = [class_values[x] for x in doc]
        peak_lists.append(list(zip(doc_bow, weights)))

    return peak_lists


def unique_peaks_linear(spectrums, min_bin_size, d_bins, mz_min):
    """Collect unique (binned) peaks."""
    unique_peaks = set()
    for spectrum in spectrums:
        for mz in bin_number_array_linear(spectrum.peaks.mz, min_bin_size, d_bins, mz_min):
            unique_peaks.add(mz)
    unique_peaks = sorted(unique_peaks)
    class_values = {}

    for i, item in enumerate(unique_peaks):
        class_values[item] = i

    return class_values, unique_peaks


def set_d_bins_linear(number, min_bin_size, mz_min=10.0, mz_max=1000.0):
    return ((mz_max - mz_min) - min_bin_size * number)/((number-1) * number/2)


def bin_number_linear(mz, min_bin_size, d_bins, mz_min=10.0):
    return (2*(mz - mz_min)/d_bins + (min_bin_size/d_bins + 0.5)**2)**0.5 - min_bin_size/d_bins - 0.5


def bin_number_array_linear(mz, min_bin_size, d_bins, mz_min=10.0):
    if d_bins != 0.0:
        bin_numbers = (2*(mz - mz_min)/d_bins + (min_bin_size/d_bins + 0.5)**2)**0.5 - min_bin_size/d_bins - 0.5
    else:
        bin_numbers = mz/min_bin_size - mz_min * min_bin_size
    return bin_numbers.astype(int)

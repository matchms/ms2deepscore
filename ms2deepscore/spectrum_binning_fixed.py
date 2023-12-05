""" Functions to create binned vector from spectrum using fixed width bins.
"""
from typing import List
import numpy as np
from matchms import Spectrum
from tqdm import tqdm


def create_peak_list_fixed(spectrums, peaks_vocab, d_bins,
                           mz_max=1000.0, mz_min=10.0, peak_scaling=0.5,
                           progress_bar=True):
    """Create list of (binned) peaks.
    
    Parameters
    ----------
    spectrums
        List of spectrums. 
    peaks_vocab
        Dictionary of all known peak bins.
    d_bins
        Bin width.
    mz_max
        Upper bound of m/z to include in binned spectrum. Default is 1000.0.
    mz_min
        Lower bound of m/z to include in binned spectrum. Default is 10.0.
    peak_scaling
        Scale all peak intensities by power pf peak_scaling. Default is 0.5.
    progress_bar
        Show progress bar if set to True. Default is True.
    """
    # pylint: disable=too-many-arguments
    peak_lists = []
    missing_fractions = []

    for spectrum in tqdm(spectrums, desc="Spectrum binning",
                         disable=(not progress_bar)):
        doc = bin_number_array_fixed(spectrum.peaks.mz, d_bins, mz_max=mz_max, mz_min=mz_min)
        weights = spectrum.peaks.intensities ** peak_scaling                
        
        # Find binned peaks present in peaks_vocab
        idx_in_vocab = [i for i, x in enumerate(doc) if x in peaks_vocab.keys()]
        idx_not_in_vocab = list(set(np.arange(len(doc))) - set(idx_in_vocab))
    
        doc_bow = [peaks_vocab[doc[i]] for i in idx_in_vocab]

        # TODO add missing weighted part!?!?
        peak_lists.append(list(zip(doc_bow, weights[idx_in_vocab])))
        if len(idx_in_vocab) == 0:
            missing_fractions.append(1.0)
        else:
            missing_fractions.append(np.sum(weights[idx_not_in_vocab])/np.sum(weights))
        
    return peak_lists, missing_fractions


def unique_peaks_fixed(spectrums: List[Spectrum], d_bins: float,
                       mz_max: float, mz_min: float):
    """Collect unique (binned) peaks.
    
    Parameters
    ----------
    spectrums
        List of spectrums. 
    d_bins
        Bin width.
    mz_max
        Upper bound of m/z to include in binned spectrum.
    mz_min
        Lower bound of m/z to include in binned spectrum.
    """
    unique_peaks = set()
    for spectrum in spectrums:
        for mz in bin_number_array_fixed(spectrum.peaks.mz, d_bins, mz_max, mz_min):
            unique_peaks.add(mz)
    unique_peaks = sorted(unique_peaks)
    class_values = {}

    for i, item in enumerate(unique_peaks):
        class_values[int(item)] = i

    return class_values, [int(x) for x in unique_peaks]


def bin_size_fixed(number, d_bins):
    return d_bins * number


def bin_number_fixed(mz, d_bins):
    """Return binned position"""
    return int(mz/d_bins)


def bin_number_array_fixed(mz_array: np.ndarray, d_bins: float,
                           mz_max: float, mz_min: float) -> np.ndarray:
    """Return binned position

    Parameters
    ----------
    mz_array
        Numpy array of peak m/z positions.
    d_bins
        Bin width.
    mz_max
        Upper bound of m/z to include in binned spectrum.
    mz_min
        Lower bound of m/z to include in binned spectrum.
    """
    mz_array_selected = mz_array[(mz_array >= mz_min) & (mz_array <= mz_max)]
    assert mz_array_selected.shape[0] > 0, "Found no peaks between mz_min and mz_max."
    bins = mz_array_selected/d_bins
    return (bins - bin_number_fixed(mz_min, d_bins)).astype(int)


def set_d_bins_fixed(number_of_bins, mz_min=10.0, mz_max=1000.0):
    return (mz_max - mz_min) / number_of_bins

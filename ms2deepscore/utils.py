import os
import pickle
from typing import Generator, List
import numba
import numpy as np
from matchms import Spectrum
from matchms.importing import load_spectra
from tqdm import tqdm


def save_pickled_file(obj, filename: str):
    if os.path.exists(filename):
        raise FileExistsError("File already exists")
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_pickled_file(filename: str):
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object


def return_non_existing_file_name(file_name):
    """Checks if a path already exists, otherwise creates a new filename with (1).
    """
    if not os.path.exists(file_name):
        return file_name
    print(f"The file name already exists: {file_name}")
    file_name_base, file_extension = os.path.splitext(file_name)
    i = 1
    new_file_name = f"{file_name_base}({i}){file_extension}"
    while os.path.exists(new_file_name):
        i += 1
        new_file_name = f"{file_name_base}({i}){file_extension}"
    print(f"Instead the file will be stored in {new_file_name}")
    return new_file_name


def load_spectra_as_list(file_name) -> List[Spectrum]:
    spectra = load_spectra(file_name, metadata_harmonization=True)
    if isinstance(spectra, Generator):
        return list(tqdm(spectra, desc="Loading in spectra"))
    return spectra


def remove_diagonal(matrix):
    """Removes the diagonal from a square matrix.
    """
    nrows, ncols = matrix.shape
    
    if nrows != ncols:
        raise ValueError("Expected a square matrix")

    strided = np.lib.stride_tricks.as_strided
    s0, s1 = nrows.strides

    return strided(matrix.ravel()[1:], shape=(nrows-1, nrows), strides=(s0 + s1, s1)).reshape(nrows, -1)


@numba.jit(nopython=True)
def scaled_intensity_sum(mz_values, intensities, min_mz=0, max_mz=1000, scaling=2):
    """Compute a scaled intensity sum for all peaks of a spectrum.

    Each peak will be scaled by the power of `scaling`.
    """
    scaled_intensity = 0
    for mz, intensity in zip(mz_values, intensities):
        if intensity > 1:
            raise ValueError("Intensities should be scaled to max. 1.")
        if min_mz <= mz <= max_mz:
            scaled_intensity += intensity ** scaling
    return scaled_intensity


def compute_scaled_intensitiy_sums(spectra, min_mz=0, max_mz=1000, scaling=2):
    """Calculates the scaled intensity sum

    This can be used to filter out badly fragmented spectra"""
    scaled_intensities = np.zeros(len(spectra))
    for i, spectrum in enumerate(spectra):
        scaled_intensities[i] = scaled_intensity_sum(
            spectrum.peaks.mz,
            spectrum.peaks.intensities,
            min_mz=min_mz,
            max_mz=max_mz,
            scaling=scaling
        )

    return scaled_intensities


def create_evenly_spaced_bins(nr_of_bins):
    """Creates evenly spaced bins between -0.0000001 and 1"""
    bin_borders = np.linspace(0, 1, nr_of_bins+1)
    bin_borders[0] = -0.00000001
    bins = [(bin_borders[i], bin_borders[i+1]) for i in range(nr_of_bins)]
    return bins


def validate_bin_order(score_bins):
    """
    Checks that the given bins are of the correct format:
    - Each bin is a tuple/list of two numbers [low, high], with low <= high
    - Bins cover the entire interval from 0 to 1, with no gaps or overlaps
    - The lowest bin starts below 0 (since pairs >=0 are selected and we want to include zero)
    """

    # Sort bins by their lower bound
    sorted_bins = sorted(score_bins, key=lambda b: b[0])

    # Check upper and lower bound
    if sorted_bins[0][0] >= 0:
        raise ValueError(f"The first bin should start below 0, but starts at {sorted_bins[0][0]}")

    if sorted_bins[-1][1] != 1:
        raise ValueError(f"The last bin should end at 1, but ends at {sorted_bins[-1][1]}")

    # Check order, format, and overlaps
    previous_high = None
    for score_bin in sorted_bins:
        if len(score_bin) != 2:
            raise ValueError("Each bin should have exactly two elements")
        low, high = score_bin
        if low > high:
            raise ValueError("The first number in the bin should be smaller than or equal to the second")
        if high < 0:
            raise ValueError("No bin should be entirely below 0.")
        if previous_high is not None:
            if low != previous_high:
                raise ValueError("There is a gap or overlap between bins; The bins should cover everything between 0 and 1.")
        previous_high = high

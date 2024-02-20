import os
import pickle
from typing import Generator, List
import numba
import numpy as np
from matchms import Spectrum
from matchms.importing import load_spectra


def save_pickled_file(obj, filename: str):
    assert not os.path.exists(filename), "File already exists"
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_pickled_file(filename: str):
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object


def return_non_existing_file_name(file_name):
    """Checks if a path already exists, otherwise creates a new filename with (1)"""
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
        return list(spectra)
    return spectra


def remove_diagonal(matrix):
    """Removes the diagonal from a matrix

    meant for removing matches of spectra against itself. """
    # Get the number of rows and columns
    nr_of_rows, nr_of_cols = matrix.shape
    if nr_of_rows != nr_of_cols:
        raise ValueError("Expected predictions against itself")

    # Create a mask for the diagonal elements
    diagonal_mask = np.eye(nr_of_rows, dtype=bool)

    # Use the mask to remove the diagonal elements
    matrix_without_diagonal = matrix[~diagonal_mask].reshape(nr_of_rows, nr_of_cols - 1)
    return matrix_without_diagonal


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

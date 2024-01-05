from typing import List
import numpy as np
from matchms.Spectrum import Spectrum
from tqdm import tqdm


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


def select_one_spectrum_per_inchikey(spectra):
    inchikeys_per_spectrum = np.array([spectrum.get("inchikey")[:14] for spectrum in spectra])
    unique_inchikeys = np.unique(inchikeys_per_spectrum)
    # Loop through unique values and pick a random index for each
    random_indices = []
    for unique_inchikey in unique_inchikeys:
        spec_idx_matching_inchikey = np.where(inchikeys_per_spectrum == unique_inchikey)[0]
        random_index = np.random.choice(spec_idx_matching_inchikey)
        random_indices.append(random_index)
    one_spectrum_per_inchikey = np.array(random_indices)
    return one_spectrum_per_inchikey


def select_predictions_symmetric(val_spectra: List[Spectrum],
                                 predicted_values,
                                 true_values):
    if not predicted_values.shape == (len(val_spectra), len(val_spectra)):
        raise ValueError("The nr of val spectra and predicted values and true values should match")
    one_spectrum_per_inchikey_idx_1 = select_one_spectrum_per_inchikey(val_spectra)
    predictions_one_spectrum_per_inchikey = predicted_values[one_spectrum_per_inchikey_idx_1, :][:,
                                            one_spectrum_per_inchikey_idx_1]
    true_values_one_spectrum_per_inchikey = true_values[one_spectrum_per_inchikey_idx_1, :][:,
                                            one_spectrum_per_inchikey_idx_1]
    return remove_diagonal(predictions_one_spectrum_per_inchikey), \
           remove_diagonal(true_values_one_spectrum_per_inchikey)


def select_predictions_not_symmetric(val_spectra: List[Spectrum],
                                     val_spectra_other_mode: List[Spectrum],
                                     predicted_values,
                                     true_values):
    if not predicted_values.shape == (len(val_spectra), len(val_spectra_other_mode)):
        raise ValueError("The nr of val spectra and predicted values and true values should match")

    one_spectrum_per_inchikey_idx_1 = select_one_spectrum_per_inchikey(val_spectra)
    one_spectrum_per_inchikey_idx_2 = select_one_spectrum_per_inchikey(val_spectra_other_mode)

    predictions_one_spectrum_per_inchikey = \
        predicted_values[one_spectrum_per_inchikey_idx_1, :][:, one_spectrum_per_inchikey_idx_2]
    true_values_one_spectrum_per_inchikey = \
        true_values[one_spectrum_per_inchikey_idx_1, :][:, one_spectrum_per_inchikey_idx_2]
    return predictions_one_spectrum_per_inchikey, true_values_one_spectrum_per_inchikey


def select_pairs_one_spectrum_per_inchikey(val_spectra: List[Spectrum],
                                           val_spectra_other_mode: List[Spectrum],
                                           predicted_values,
                                           true_values):
    """Selects the predicted values and true values for one randomly picked spectrum per inchikey

    If the validation spectra are the same (comparison against itself) the spectra that match against itself are removed
    Matches between the same inchikey are allowed. If it is not symmetric there is no risk of matching against itself."""
    if not predicted_values.shape == true_values.shape:
        raise ValueError("The shape of the predicted values and the true values should match")
    is_symmetric = (val_spectra == val_spectra_other_mode)

    if is_symmetric:
        selected_predictions, selected_true_values = \
            select_predictions_symmetric(val_spectra, predicted_values, true_values)
    else:
        selected_predictions, selected_true_values = \
            select_predictions_not_symmetric(val_spectra, val_spectra_other_mode, predicted_values, true_values)
    return selected_predictions, selected_true_values


def sample_spectra_multiple_times(val_spectra: List[Spectrum],
                                  val_spectra_other_mode: List[Spectrum],
                                  predicted_values: np.array,
                                  true_values: np.array,
                                  nr_of_sample_times: int):
    combined_predictions, combined_true_values = \
        select_pairs_one_spectrum_per_inchikey(val_spectra, val_spectra_other_mode, predicted_values, true_values)
    for _ in tqdm(range(nr_of_sample_times),
                  desc="Sampling 1 spectrum per inchikey (multiple times)"):
        predictions_one_spectrum_per_inchikey, true_values_one_spectrum_per_inchikey = \
            select_pairs_one_spectrum_per_inchikey(val_spectra, val_spectra_other_mode, predicted_values, true_values)
        combined_predictions = np.concatenate((combined_predictions, predictions_one_spectrum_per_inchikey), axis=0)
        combined_true_values = np.concatenate((combined_true_values, true_values_one_spectrum_per_inchikey), axis=0)
    return combined_true_values, combined_predictions

import numpy as np
from tqdm import tqdm


def remove_diagonal(matrix):
    # Get the number of rows and columns
    nr_of_rows, nr_of_cols = matrix.shape
    if nr_of_rows != nr_of_cols:
        raise ValueError("Expected predictions against itself")

    # Create a mask for the diagonal elements
    diagonal_mask = np.eye(nr_of_rows, dtype=bool)

    # Use the mask to remove the diagonal elements
    matrix_without_diagonal = matrix[~diagonal_mask].reshape(nr_of_rows, nr_of_cols - 1)
    return matrix_without_diagonal


def select_one_spectrum_per_inchikey(val_spectra,
                                     predicted_values,
                                     true_values):
    """Selects the predicted values and true values for one randomly picked spectrum per inchikey"""
    list_of_inchikeys = []
    for spectrum in val_spectra:
        inchikey = spectrum.get("inchikey")[:14]
        list_of_inchikeys.append(inchikey)
    inchikeys_per_spectrum = np.array(list_of_inchikeys)
    unique_inchikeys = np.unique(inchikeys_per_spectrum)
    # Loop through unique values and pick a random index for each
    random_indices = []
    for i, value in enumerate(unique_inchikeys):
        spec_idx_matching_inchikey = np.where(inchikeys_per_spectrum == value)[0]
        random_index = np.random.choice(spec_idx_matching_inchikey)
        random_indices.append(random_index)
    one_spectrum_per_inchikey = np.array(random_indices)
    predictions_one_spectrum_per_inchikey = predicted_values[one_spectrum_per_inchikey, :][:, one_spectrum_per_inchikey]
    true_values_one_spectrum_per_inchikey = true_values[one_spectrum_per_inchikey, :][:, one_spectrum_per_inchikey]
    return remove_diagonal(predictions_one_spectrum_per_inchikey), remove_diagonal(true_values_one_spectrum_per_inchikey)


def sample_spectra_multiple_times(val_spectra,
                                  predicted_values,
                                  true_values,
                                  nr_of_sample_times: int):
    combined_predictions, combined_true_values = \
        select_one_spectrum_per_inchikey(val_spectra, predicted_values, true_values)
    for _ in tqdm(range(nr_of_sample_times)):
        predictions_one_spectrum_per_inchikey, true_values_one_spectrum_per_inchikey = \
            select_one_spectrum_per_inchikey(val_spectra, predicted_values, true_values)
        combined_predictions = np.concatenate((combined_predictions, predictions_one_spectrum_per_inchikey), axis=0)
        combined_true_values = np.concatenate((combined_true_values, true_values_one_spectrum_per_inchikey), axis=0)
    return combined_true_values, combined_predictions

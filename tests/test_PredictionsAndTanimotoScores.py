import string

import numpy as np
import pandas as pd

from ms2deepscore.models.loss_functions import LOSS_FUNCTIONS
from ms2deepscore.utils import create_evenly_spaced_bins
from ms2deepscore.validation_loss_calculation.PredictionsAndTanimotoScores import PredictionsAndTanimotoScores
import random


def dummy_tanimoto_scores(num_of_unique_inchikeys, num_of_unique_inchikeys2):
    """Creates a dataframe with dummy tanimoto scores"""
    if num_of_unique_inchikeys > 26 or num_of_unique_inchikeys2 > 26:
        raise ValueError("Maximum number of unique inchikeys is 26 (numer of letters in alphabet), "
                         "if you need more, change the code below")
    # Create list of dummy_inchikeys, with the same letter repeating
    dummy_inchikeys = [f"{14 * letter}" for letter in list(string.ascii_uppercase[:num_of_unique_inchikeys])]
    dummy_inchikeys_2 = [f"{14 * letter}" for letter in list(string.ascii_uppercase[:num_of_unique_inchikeys2])]

    # Generate random values
    random_values = np.random.rand(len(dummy_inchikeys), len(dummy_inchikeys_2))
    # Create the DataFrame with tanimoto_scores
    tanimoto_scores = pd.DataFrame(random_values, index=dummy_inchikeys, columns=dummy_inchikeys_2)

    # Make the matrix symmetric
    for inchikey in dummy_inchikeys:
        for inchikey_2 in dummy_inchikeys_2:
            if inchikey == inchikey_2:
                tanimoto_scores[inchikey][inchikey_2] = 1.0
            else:
                try:
                    tanimoto_scores[inchikey][inchikey_2] = tanimoto_scores[inchikey_2][inchikey]
                except KeyError:
                    continue
    return tanimoto_scores


def create_dummy_predictions_symmetric(tanimoto_scores: pd.DataFrame):
    """Creates a dataframe with predictions that has the same inchikeys as the given tanimoto scores,
    but for half of the tanimoto scores extra colums are added to mimick multiple spectra per inchikey"""
    predictions = tanimoto_scores.__deepcopy__()
    if predictions.shape[0] != predictions.shape[1]:
        raise ValueError("Predictions are expected to not be symmetric")
    inchikeys = list(predictions.index)
    random_half_of_inchikeys = random.sample(inchikeys, len(inchikeys)//2)
    for inchikey in random_half_of_inchikeys:
        new_row = tanimoto_scores.loc[inchikey].copy()
        new_row.name = inchikey  # Set the name for the new row
        predictions = pd.concat([predictions, pd.DataFrame([new_row])])
    for inchikey in random_half_of_inchikeys:
        predictions["new_copy"] = predictions[inchikey]
        predictions.rename(columns={'new_copy': inchikey}, inplace=True)
    random_noise = np.random.normal(loc=0, scale=0.1, size=predictions.shape)
    predictions = predictions + random_noise
    symmetric_predictions = (predictions + predictions.T) / 2
    # Set diagonal to nan
    for i in range(min(symmetric_predictions.shape)):
        symmetric_predictions.iloc[i, i] = None
    return symmetric_predictions


def create_dummy_predictions_not_symmetric(tanimoto_scores: pd.DataFrame):
    """Creates a dataframe with predictions that has the same inchikeys as the given tanimoto scores,
    but for half of the tanimoto scores extra colums are added to mimick multiple spectra per inchikey"""
    predictions = tanimoto_scores.__deepcopy__()
    if predictions.shape[0] == predictions.shape[1]:
        raise ValueError("Predictions are expected to not be symmetric")
    inchikeys = list(predictions.index)
    # Randomly duplicate half of the rows
    for inchikey in random.sample(inchikeys, len(inchikeys)//2):
        new_row = tanimoto_scores.loc[inchikey].copy()
        new_row.name = inchikey  # Set the name for the new row
        predictions = pd.concat([predictions, pd.DataFrame([new_row])])
    # Randomly duplicate half of the columns
    inchikeys_2 = list(predictions.columns)
    for inchikey in random.sample(inchikeys_2, len(inchikeys_2)//2):
        predictions["new_copy"] = predictions[inchikey]
        predictions.rename(columns={'new_copy': inchikey}, inplace=True)
    random_noise = np.random.normal(loc=0, scale=0.1, size=predictions.shape)
    predictions = predictions + random_noise
    # Set diagonal to nan
    for i in range(min(tanimoto_scores.shape)):
        predictions.iloc[i, i] = None
    return predictions


def create_dummy_predictions_and_tanimoto_scores(nr_of_unique_inchikeys, nr_of_unique_inchikeys_2):
    """Creates a dummy dataset"""
    tanimoto_scores = dummy_tanimoto_scores(nr_of_unique_inchikeys, nr_of_unique_inchikeys_2)
    if nr_of_unique_inchikeys == nr_of_unique_inchikeys_2:
        predictions = create_dummy_predictions_symmetric(tanimoto_scores)
    else:
        predictions = create_dummy_predictions_not_symmetric(tanimoto_scores)
    symmetric = nr_of_unique_inchikeys == nr_of_unique_inchikeys_2
    return PredictionsAndTanimotoScores(predictions, tanimoto_scores, symmetric)


def test_predictions_and_tanimoto_scores():
    """Just checks that the functions run and have the expected formats"""
    matrix_dimensions = (26, 10)
    predictions_and_tanimoto_scores = create_dummy_predictions_and_tanimoto_scores(matrix_dimensions[0], matrix_dimensions[1])
    average_prediction = predictions_and_tanimoto_scores.get_average_prediction_per_inchikey_pair()
    assert isinstance(average_prediction, pd.DataFrame)
    assert np.all(predictions_and_tanimoto_scores.tanimoto_df.index == average_prediction.index)
    assert np.all(predictions_and_tanimoto_scores.tanimoto_df.columns == average_prediction.columns)
    for loss_function in LOSS_FUNCTIONS.keys():
        nr_of_bins = 3
        bin_content, average_loss_per_bin = predictions_and_tanimoto_scores.get_average_loss_per_bin_per_inchikey_pair(
            loss_function, create_evenly_spaced_bins(nr_of_bins))
        assert len(bin_content) == len(average_loss_per_bin) == nr_of_bins
        assert sum(bin_content) == matrix_dimensions[0] * matrix_dimensions[1]

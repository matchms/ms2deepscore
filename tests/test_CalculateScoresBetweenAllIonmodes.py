import string

import numpy as np
import pandas as pd

from ms2deepscore.benchmarking.CalculateScoresBetweenAllIonmodes import CalculateScoresBetweenAllIonmodes, PredictionsAndTanimotoScores
from tests.test_user_worfklow import load_processed_spectrums, TEST_RESOURCES_PATH
import random


def dummy_tanimoto_scores(num_of_unique_inchikeys):
    """Creates a dataframe with dummy tanimoto scores"""
    # Create list of dummy_inchikeys, with the same letter repeating
    dummy_inchikeys = [f"{14 * letter}" for letter in list(string.ascii_uppercase[:num_of_unique_inchikeys])]
    # Generate random values
    random_values = np.random.rand(len(dummy_inchikeys), len(dummy_inchikeys))
    # Make the matrix symmetric
    symmetric_values = (random_values + random_values.T) / 2
    # Create the DataFrame with tanimoto_scores
    tanimoto_scores = pd.DataFrame(symmetric_values, index=dummy_inchikeys, columns=dummy_inchikeys)
    # Make sure all the diagonals are 1
    np.fill_diagonal(tanimoto_scores.values, 1)
    return tanimoto_scores


def create_dummy_predictions(tanimoto_scores: pd.DataFrame):
    """Creates a dataframe with predictions that has the same inchikeys as the given tanimoto scores,
    but for half of the tanimoto scores extra colums are added to mimick multiple spectra per inchikey"""
    predictions = tanimoto_scores.__deepcopy__()
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
    return symmetric_predictions


def create_dummy_predictions_and_tanimoto_scores(nr_of_unique_inchikeys):
    tanimoto_scores = dummy_tanimoto_scores(nr_of_unique_inchikeys)
    predictions = create_dummy_predictions(tanimoto_scores)
    return PredictionsAndTanimotoScores(predictions, tanimoto_scores, True)


def test_predictions_and_tanimoto_scores():
    predictions_and_tanimoto_scores = create_dummy_predictions_and_tanimoto_scores(26)
    # todo add actual tests for correctly creating other files


def test_calculate_scores_between_all_ionmodes():
    spectrums = load_processed_spectrums()

    # Load pretrained model
    model_file = TEST_RESOURCES_PATH / "testmodel.pt"
    scores_between_all_ionmodes = CalculateScoresBetweenAllIonmodes(model_file, spectrums, spectrums)
    # todo add actual tests that scores are calculated correctly
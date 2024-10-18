import string

import numpy as np
import pandas as pd
import pytest

from ms2deepscore.benchmarking.CalculateScoresBetweenAllIonmodes import CalculateScoresBetweenAllIonmodes, PredictionsAndTanimotoScores
from ms2deepscore.benchmarking.plot_heatmaps import create_3_heatmaps
from tests.test_user_worfklow import load_processed_spectrums, TEST_RESOURCES_PATH
import random
from ms2deepscore.benchmarking.plot_average_per_bin import plot_average_per_bin


def dummy_tanimoto_scores(num_of_unique_inchikeys):
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
    print(scores_between_all_ionmodes)

@pytest.fixture()
def dummy_scores():
    def create_test_scores(num_tuples, noise_std=0.1):
        first_numbers = []
        second_numbers = []
        for _ in range(num_tuples):
            first_number = random.uniform(0, 1)  # First number between 0 and 1
            noise = random.gauss(0, noise_std)  # Random noise with mean 0 and standard deviation `noise_std`
            second_number = first_number + noise  # Second number based on the first with noise
            first_numbers.append(first_number)
            second_numbers.append(second_number)
        return second_numbers, first_numbers

    class TestPredictionsAndTanimotoScores(PredictionsAndTanimotoScores):
        def __init__(self):
            nr_of_pairs = 10000
            random.seed(42)
            self.list_of_average_predictions, self.list_of_tanimoto_scores = create_test_scores(nr_of_pairs)

    class TestCalculateScoresBetweenAllIonmodes(CalculateScoresBetweenAllIonmodes):
        def __init__(self):
            self.pos_vs_neg_scores = TestPredictionsAndTanimotoScores()
            self.pos_vs_pos_scores = TestPredictionsAndTanimotoScores()
            self.neg_vs_neg_scores = TestPredictionsAndTanimotoScores()
    return TestCalculateScoresBetweenAllIonmodes()


def test_create_three_heatmaps(dummy_scores):
    fig = create_3_heatmaps(dummy_scores, 50)
    fig.show()


def test_plot_average_per_bin(dummy_scores):
    fig = plot_average_per_bin(dummy_scores, 50)
    fig.show()

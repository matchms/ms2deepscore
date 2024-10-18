import pytest

from ms2deepscore.benchmarking.CalculateScoresBetweenAllIonmodes import CalculateScoresBetweenAllIonmodes
from ms2deepscore.benchmarking.plot_heatmaps import create_3_heatmaps
from tests.test_user_worfklow import load_processed_spectrums, TEST_RESOURCES_PATH
import random
from ms2deepscore.benchmarking.plot_average_per_bin import plot_average_per_bin


def testCreatePlots():
    spectrums = load_processed_spectrums()

    # Load pretrained model
    model_file = TEST_RESOURCES_PATH / "testmodel.pt"
    plots_creator = CalculateScoresBetweenAllIonmodes(model_file, spectrums, spectrums)


@pytest.fixture()
def dummy_pairs():
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

    class TestInstance(CalculateScoresBetweenAllIonmodes):
        def __init__(self):
            nr_of_pairs = 10000
            random.seed(42)
            self.pos_vs_neg_predictions, self.pos_vs_neg_tanimoto_scores = create_test_scores(nr_of_pairs)
            self.pos_vs_pos_predictions, self.pos_vs_pos_tanimoto_scores = create_test_scores(nr_of_pairs)
            self.neg_vs_neg_predictions, self.neg_vs_neg_tanimoto_scores = create_test_scores(nr_of_pairs)
    return TestInstance()


def test_create_three_heatmaps(dummy_pairs):
    fig = create_3_heatmaps(dummy_pairs, 50)
    fig.show()


def test_plot_average_per_bin(dummy_pairs):
    fig = plot_average_per_bin(dummy_pairs, 50)
    fig.show()

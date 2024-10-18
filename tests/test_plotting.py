import random
import pytest
import numpy as np

from ms2deepscore.benchmarking.CalculateScoresBetweenAllIonmodes import PredictionsAndTanimotoScores, \
    CalculateScoresBetweenAllIonmodes
from ms2deepscore.benchmarking.plot_average_per_bin import plot_average_per_bin
from ms2deepscore.benchmarking.plot_heatmaps import create_3_heatmaps
from ms2deepscore.benchmarking.plot_rmse_per_bin import (
    plot_rmse_per_bin, plot_rmse_per_bin_multiple_benchmarks)
from ms2deepscore.benchmarking.plot_stacked_histogram import (
    calculate_all_histograms, plot_reversed_stacked_histogram_plot,
    plot_stacked_histogram_plot_wrapper)
from ms2deepscore.benchmarking.plotting import create_confusion_matrix_plot


def test_create_confusion_matrix_plot():
    fig = create_confusion_matrix_plot(np.random.random((100, 100)), np.random.random((100, 100)))
    assert fig is not None
    assert fig.get_axes()[0].get_xlabel() == "MS2DeepScore"
    assert fig.get_axes()[0].get_ylabel() == 'Tanimoto similarity'


def test_calculate_histograms():
    nr_of_bins = 5
    tanimoto_bins = np.linspace(0, 1, nr_of_bins + 1)
    tanimoto_bins[-1] = 1.0000000001
    normalized_counts_per_bin, used_ms2deepscore_bins_per_bin, percentage_of_total_pairs_per_bin = \
        calculate_all_histograms(np.random.random((100, 100)), np.random.random((100, 100)), tanimoto_bins)

    # Ensure the number of histograms, used bins and bin contents are the same
    assert len(normalized_counts_per_bin) == len(used_ms2deepscore_bins_per_bin) == \
           len(percentage_of_total_pairs_per_bin) == nr_of_bins

    # Ensure the used bins are valid
    for ms2deepscore_bins in used_ms2deepscore_bins_per_bin:
        assert np.all(sorted(ms2deepscore_bins) == ms2deepscore_bins)
        assert round(ms2deepscore_bins[0]) == 0
        assert round(ms2deepscore_bins[-1]) == 1
    # Ensure histograms are properly formed
    for i in range(len(normalized_counts_per_bin)):
        assert len(normalized_counts_per_bin[i]) == len(
            used_ms2deepscore_bins_per_bin[i]) - 1  # histogram frequencies and bin edges


def test_plot_histograms():
    np.random.seed(123)
    dimension = (100, 100)
    plot_stacked_histogram_plot_wrapper(np.random.random(dimension) ** 2,
                                        np.random.random(dimension) ** 2,
                                        n_bins=10)


def test_reverse_plot_stacked_histogram():
    np.random.seed(123)
    dimension = (100, 100)
    plot_reversed_stacked_histogram_plot(np.random.random(dimension) ** 2,
                                         np.random.random(dimension) ** 2)


def test_plot_rmse_per_bin():
    plot_rmse_per_bin(predicted_scores=np.random.random((200, 200)) ** 3,
                      true_scores=np.random.random((200, 200)) ** 3)


def test_plot_rmse_per_bin_multiple_benchmarks():
    plot_rmse_per_bin_multiple_benchmarks(
        [np.random.random((200, 200)) ** 3, np.random.random((100, 100)) ** 1,
         np.random.random((200, 100)) ** 1, np.random.random((300, 300)) ** 1, ],
        [np.random.random((200, 200)) ** 3, np.random.random((100, 100)) ** 3,
         np.random.random((200, 100)) ** 2, np.random.random((300, 300)) ** 2, ],
        ["positive vs positive", 'negative vs negative',
         "positive vs negative", "both vs both"])


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

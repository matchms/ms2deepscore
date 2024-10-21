import random
import pytest
import numpy as np
from matplotlib import pyplot as plt

from ms2deepscore.benchmarking.CalculateScoresBetweenAllIonmodes import PredictionsAndTanimotoScores, \
    CalculateScoresBetweenAllIonmodes
from ms2deepscore.benchmarking.plot_average_per_bin import plot_average_per_bin
from ms2deepscore.benchmarking.plot_heatmaps import create_3_heatmaps
from ms2deepscore.benchmarking.plot_rmse_per_bin import (
    plot_loss_per_bin, plot_loss_per_bin_multiple_benchmarks)
from ms2deepscore.benchmarking.plot_stacked_histogram import (
    calculate_all_histograms, plot_reversed_stacked_histogram_plot,
    plot_stacked_histogram_plot_wrapper)
from ms2deepscore.benchmarking.plotting import create_confusion_matrix_plot
from tests.test_CalculateScoresBetweenAllIonmodes import create_dummy_predictions_and_tanimoto_scores


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


@pytest.fixture()
def scores_between_all_ionmodes():
    class TestCalculateScoresBetweenAllIonmodes(CalculateScoresBetweenAllIonmodes):
        def __init__(self):
            self.pos_vs_neg_scores = create_dummy_predictions_and_tanimoto_scores(26, 14)
            self.pos_vs_pos_scores = create_dummy_predictions_and_tanimoto_scores(26, 26)
            self.neg_vs_neg_scores = create_dummy_predictions_and_tanimoto_scores(14, 14)
    return TestCalculateScoresBetweenAllIonmodes()


def test_create_three_heatmaps(scores_between_all_ionmodes):
    fig = create_3_heatmaps(scores_between_all_ionmodes, 30)
    fig.show()


def test_plot_average_per_bin(scores_between_all_ionmodes):
    fig = plot_average_per_bin(scores_between_all_ionmodes, 10)
    fig.show()


def test_plot_loss_per_bin_multiple_benchmarks(scores_between_all_ionmodes):
    plot_loss_per_bin_multiple_benchmarks(scores_between_all_ionmodes.list_of_predictions_and_tanimoto_scores(),
                                          loss_type="MSE")
    plt.show()
    plot_loss_per_bin_multiple_benchmarks(scores_between_all_ionmodes.list_of_predictions_and_tanimoto_scores(),
                                          loss_type="RMSE")
    plt.show()
    plot_loss_per_bin_multiple_benchmarks(scores_between_all_ionmodes.list_of_predictions_and_tanimoto_scores(),
                                          loss_type="MAE")
    plt.show()


def test_plot_loss_per_bin():
    predictions_and_tanimoto_scores = create_dummy_predictions_and_tanimoto_scores(26, 14)
    plot_loss_per_bin(predictions_and_tanimoto_scores, loss_type="MSE")
    plt.show()
    plot_loss_per_bin(predictions_and_tanimoto_scores, loss_type="RMSE")
    plt.show()
    plot_loss_per_bin(predictions_and_tanimoto_scores, loss_type="MAE")
    plt.show()

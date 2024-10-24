import pytest
from matplotlib import pyplot as plt

from ms2deepscore.benchmarking.CalculateScoresBetweenAllIonmodes import CalculateScoresBetweenAllIonmodes
from ms2deepscore.benchmarking.plot_average_per_bin import plot_average_per_bin
from ms2deepscore.benchmarking.plot_heatmaps import create_3_heatmaps
from ms2deepscore.benchmarking.plot_rmse_per_bin import (
    plot_loss_per_bin, plot_loss_per_bin_multiple_benchmarks)
from tests.test_PredictionsAndTanimotoScores import create_dummy_predictions_and_tanimoto_scores


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

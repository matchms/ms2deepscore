import pytest

from ms2deepscore.benchmarking.CalculateScoresBetweenAllIonmodes import CalculateScoresBetweenAllIonmodes
from ms2deepscore.benchmarking.plot_average_per_bin import plot_average_per_bin
from ms2deepscore.benchmarking.plot_heatmaps import create_3_heatmaps
from ms2deepscore.benchmarking.plot_loss_per_bin import (
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
    create_3_heatmaps(scores_between_all_ionmodes, 30)


def test_plot_average_per_bin(scores_between_all_ionmodes):
    plot_average_per_bin(scores_between_all_ionmodes, 10)


def test_plot_loss_per_bin_multiple_benchmarks(scores_between_all_ionmodes):
    plot_loss_per_bin_multiple_benchmarks(scores_between_all_ionmodes.list_of_predictions_and_tanimoto_scores(),
                                          loss_type="MSE")
    plot_loss_per_bin_multiple_benchmarks(scores_between_all_ionmodes.list_of_predictions_and_tanimoto_scores(),
                                          loss_type="RMSE")
    plot_loss_per_bin_multiple_benchmarks(scores_between_all_ionmodes.list_of_predictions_and_tanimoto_scores(),
                                          loss_type="MAE")


def test_plot_loss_per_bin():
    predictions_and_tanimoto_scores = create_dummy_predictions_and_tanimoto_scores(26, 14)
    plot_loss_per_bin(predictions_and_tanimoto_scores, loss_type="MSE")
    plot_loss_per_bin(predictions_and_tanimoto_scores, loss_type="RMSE")
    plot_loss_per_bin(predictions_and_tanimoto_scores, loss_type="MAE")

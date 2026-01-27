import os
from matplotlib import pyplot as plt

from ms2deepscore.benchmarking.CalculateScoresBetweenAllIonmodes import CalculateScoresBetweenAllIonmodes
from ms2deepscore.benchmarking.plot_average_per_bin import plot_average_per_bin
from ms2deepscore.benchmarking.plot_heatmaps import create_3_heatmaps
from ms2deepscore.benchmarking.plot_loss_per_bin import plot_loss_per_bin_multiple_benchmarks


def create_plots_between_ionmodes(scores_between_all_ionmodes: CalculateScoresBetweenAllIonmodes,
                                  results_folder: str,
                                  nr_of_bins=50):
    """
    Creates comparison plots between different ion modes.

    Parameters
    ----------
    scores_between_all_ionmodes:
        CalculateScoresBetweenAllIonmodes object containing predictions and tanimoto scores between ionmodes.
    results_folder:
        Directory where the plots will be saved.
    nr_of_bins:
        Nr of tanimoto score bins to use for plotting.
    """
    os.makedirs(results_folder, exist_ok=True)

    plot_loss_per_bin_multiple_benchmarks(scores_between_all_ionmodes.list_of_predictions_and_tanimoto_scores(),
                                          nr_of_bins=nr_of_bins,
                                          loss_type="RMSE")
    plt.savefig(os.path.join(results_folder, "RMSE_per_bin.svg"))

    plot_loss_per_bin_multiple_benchmarks(scores_between_all_ionmodes.list_of_predictions_and_tanimoto_scores(),
                                          nr_of_bins=nr_of_bins,
                                          loss_type="MSE")
    plt.savefig(os.path.join(results_folder, "MSE_per_bin.svg"))

    plot_loss_per_bin_multiple_benchmarks(scores_between_all_ionmodes.list_of_predictions_and_tanimoto_scores(),
                                          nr_of_bins=nr_of_bins,
                                          loss_type="MAE")
    plt.savefig(os.path.join(results_folder, "MAE_per_bin.svg"))

    fig = create_3_heatmaps(scores_between_all_ionmodes, nr_of_bins)
    fig.savefig(os.path.join(results_folder, "heatmaps.svg"))

    fig = plot_average_per_bin(scores_between_all_ionmodes, nr_of_bins)
    fig.savefig(os.path.join(results_folder, "average_per_bin.svg"))

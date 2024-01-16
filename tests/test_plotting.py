import numpy as np
from ms2deepscore.benchmarking_models.plot_stacked_histogram import (
    calculate_all_histograms, plot_reversed_stacked_histogram_plot,
    plot_stacked_histogram_plot_wrapper)
from ms2deepscore.benchmarking_models.plotting import \
    create_confusion_matrix_plot
from ms2deepscore.benchmarking_models.plot_rmse_per_bin import plot_rmse_per_bin_multiple_benchmarks, plot_rmse_per_bin


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
